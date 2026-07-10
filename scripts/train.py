"""Main training entrypoint for MuZero."""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import get_config
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork
from src.training.run_id import generate_run_id
from src.training.trainer import MuZeroTrainer


GAMES = {
    "tictactoe": TicTacToe,
}

# Lazy imports for games that may not be needed
def get_game(name: str):
    if name in GAMES:
        return GAMES[name]()
    if name == "connect4":
        from src.games.connect4 import Connect4
        return Connect4()
    if name in ("chess", "chess_small", "chess_hybrid", "chess_hybrid_xl"):
        from src.games.chess import ChessGame
        return ChessGame()
    if name == "checkers":
        from src.games.checkers import Checkers
        return Checkers()
    raise ValueError(f"Unknown game: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train MuZero")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "connect4", "chess", "chess_small",
                                 "chess_hybrid", "chess_hybrid_xl", "checkers"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: auto-generate YYYY_MM_DD_NNNN). "
                             "Pass an existing ID to continue writing into that run's dirs.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--warmstart-body", type=str, default=None,
                        help="Warm-start the body (+ shape-matching heads) from a checkpoint whose "
                             "architecture differs (e.g. after widening the moves-left head). Loads model "
                             "weights non-strict (changed heads keep fresh init), keeps the buffer and step, "
                             "fresh optimizer. Mutually exclusive with --resume.")
    parser.add_argument("--sample-k", type=int, default=None,
                        help="Sampled MuZero K. None = deterministic top-K (legacy).")
    parser.add_argument("--use-gumbel", action="store_true",
                        help="Enable Plain Gumbel MuZero (Danihelka 2022) at root.")
    parser.add_argument("--gumbel-m", type=int, default=None,
                        help="gumbel_num_considered (m for Sequential Halving).")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Override config.eval_interval (steps between evals).")
    parser.add_argument("--root-heavy-loss", action="store_true",
                        help="Weight the root prediction at 1.0 and each unroll step "
                             "at 1/K (MuZero paper / muzero-general convention). "
                             "Default is uniform 1/(K+1) per step (LightZero-like).")
    parser.add_argument("--max-buf-save-games", type=int, default=None,
                        help="Cap the number of most-recent self-play games persisted "
                             "to .buf per checkpoint. In-memory buffer is unaffected. "
                             "Default: no cap.")
    parser.add_argument("--stockfish-injection-path", type=str, default=None,
                        help="Directory or glob of .pkl shards (list[GameHistory]) "
                             "to inject into the buffer as self-play surrogates. "
                             "Shards are consumed in sorted-path order; cursor persists "
                             "across resume. Self-play + reanalyze auto-gate off until "
                             "the pool exhausts, then flip on automatically.")
    parser.add_argument("--stockfish-injection-games", type=int, default=None,
                        help="Games per injection round. Overrides config.")
    parser.add_argument("--stockfish-injection-interval", type=int, default=None,
                        help="Training steps between injection rounds. Overrides config.")
    parser.add_argument("--stockfish-injection-shuffle-seed", type=int, default=0,
                        help="Seed for the deterministic shuffle of injection shard paths. "
                             "0 = shuffle with seed 0 (interleaves buckets/workers so the "
                             "replay buffer sees a balanced mix of sub-pools throughout "
                             "warmstart, instead of bucket-sequential curriculum order). "
                             "Pass a negative value to disable shuffling (consume in sorted "
                             "alphabetical order — legacy behavior, reproducible across resumes).")
    parser.add_argument("--reset-injection-cursor", action="store_true",
                        help="After --resume, reset the Stockfish injection cursor to 0 so "
                             "the warmstart pool re-fills from the start. Required when "
                             "resuming into a checkpoint whose injection_loaded=pool_size "
                             "but you want lever-2 (warmstart_sample_frac) to actually "
                             "have warmstart games to anchor on. The model is allowed to "
                             "see the same teacher games multiple times across resumes — "
                             "that is the point of the anchor.")
    parser.add_argument("--num-simulations", type=int, default=None,
                        help="Override config.num_simulations.")
    parser.add_argument("--num-parallel-games", type=int, default=None,
                        help="Override config.num_parallel_games.")
    parser.add_argument("--num-self-play-games", type=int, default=None,
                        help="Override config.num_self_play_games (games per self-play phase).")
    parser.add_argument("--self-play-interval", type=int, default=None,
                        help="Override config.self_play_interval (training steps between "
                             "self-play rounds). Scale with num_self_play_games to hold the "
                             "reuse ratio (batch*interval)/(games*avg_len) constant.")
    parser.add_argument("--warmstart-sample-frac", type=float, default=None,
                        help="Override config.warmstart_sample_frac. Set to 0.0 "
                             "for a pure-self-play run with no warmstart anchor "
                             "(also pass --stockfish-injection-games 0 + "
                             "--stockfish-injection-interval 0 to disable injection).")
    parser.add_argument("--warmstart-sample-frac-final", type=float, default=None,
                        help="Override config.warmstart_sample_frac_final: the end value "
                             "the warmstart anchor anneals to (with --warmstart-anneal-frac).")
    parser.add_argument("--warmstart-anneal-frac", type=float, default=None,
                        help="Override config.warmstart_anneal_frac: fraction of training over "
                             "which warmstart_sample_frac decays to its final value. 0 = constant.")
    parser.add_argument("--self-play-warmup-steps", type=int, default=None,
                        help="Override config.self_play_warmup_steps. Two-phase "
                             "(Option A) curriculum: self-play + reanalyze are gated "
                             "OFF for this many steps (pure supervised Stockfish "
                             "warmstart pretrain), then flip on for the rest of "
                             "training. The warmstart anchor persists into phase 2. "
                             "0 = legacy pool-exhaustion gate.")
    parser.add_argument("--self-play-warmup-frac", type=float, default=None,
                        help="Fractional form of --self-play-warmup-steps (fraction "
                             "of training_steps), for notation consistency with "
                             "--batch-mixture-schedule. Resolves to "
                             "round(frac * training_steps); takes precedence over "
                             "--self-play-warmup-steps if both are given. Keeps the "
                             "self-play boundary aligned with the mixture's self-play "
                             "onset when training_steps changes.")
    parser.add_argument("--warmstart-q-ratio", type=float, default=None,
                        help="Override config.warmstart_q_ratio. Weight on the "
                             "GAME-OUTCOME blend into the Stockfish-eval WDL target "
                             "for warmstart positions (external teacher → safe to run "
                             "hot, e.g. 0.5).")
    parser.add_argument("--selfplay-q-ratio", type=float, default=None,
                        help="Override config.selfplay_q_ratio. Weight on the "
                             "MCTS-root-value blend into the outcome one-hot for "
                             "self-play positions (self-referential → keep cool, "
                             "e.g. 0.1; AlphaZero/Lc0 default 0.0).")
    parser.add_argument("--temperature-drop-step", type=int, default=None,
                        help="Override config.temperature_drop_step (plies of tau=init before tau=final).")
    parser.add_argument("--dirichlet-alpha", type=float, default=None,
                        help="Override config.dirichlet_alpha (root exploration noise concentration).")
    parser.add_argument("--no-moves-left", dest="use_moves_left", action="store_false", default=None,
                        help="Disable the moves-left head AND its MCTS utility for this run (ablation).")
    parser.add_argument("--use-moves-left", dest="use_moves_left", action="store_true", default=None,
                        help="Force-enable the moves-left head.")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Override config.repetition_penalty. δ in [0,1]: tilts "
                             "the value target of a self-play NO-PROGRESS draw "
                             "(threefold repetition OR 75-move rule) from [0,1,0] to "
                             "[0,1-δ,δ] (Draw→Loss mass), teaching the value head that "
                             "shuffling with no progress is mildly bad. Stalemate / "
                             "insufficient-material / ply-cap draws are untouched. "
                             "0.0 (default) = off / legacy behavior.")
    parser.add_argument("--repetition-penalty-window", type=int, default=None,
                        help="Override config.repetition_penalty_window. >0 ramps the "
                             "repetition δ tilt by proximity to the draw: full δ at the "
                             "terminal drawn position, linearly →0 for plies >= window "
                             "before it (per-ply shuffle-depth credit assignment). "
                             "0 (default) = uniform δ on every ply (legacy).")
    parser.add_argument("--repetition-penalty-decay", type=float, default=None,
                        help="Override config.repetition_penalty_decay (γ). >0 weights "
                             "the repetition δ by γ**plies_to_end — full δ at the drawn "
                             "position, geometric soft-tail decay backward (discount-style; "
                             "takes precedence over --repetition-penalty-window). "
                             "γ=0.93 ≈ half-strength 9.5 plies back. 0 (default) = inactive.")
    parser.add_argument("--draw-score", type=float, default=None,
                        help="Override config.draw_score. WDL→scalar value uses "
                             "V = P(W) - P(L) + draw_score·P(D); more-negative subtracts the "
                             "draw mass that squashes won-but-unconverted positions to V≈0, so "
                             "a converting move can out-rank a safe shuffle in MCTS. Preset -0.05.")
    parser.add_argument("--eval-to-wdl-alpha", type=float, default=None,
                        help="Override config.eval_to_wdl_alpha. Logistic slope mapping a scalar "
                             "eval/root_value → (P_W,P_D,P_L); larger = sharper (narrower draw "
                             "zone → more decisive WDL targets). Preset 4.0.")
    parser.add_argument("--eval-to-wdl-beta", type=float, default=None,
                        help="Override config.eval_to_wdl_beta. Draw-width of the eval→WDL "
                             "logistic; smaller = narrower draw zone (more decisive). Preset 2.0.")
    parser.add_argument("--decisive-sample-frac", type=float, default=None,
                        help="Override config.decisive_sample_frac. Fraction of each batch "
                             "force-drawn from DECISIVE self-play games (|game_outcome|=1) so the "
                             "value head can't collapse to draw-everywhere. Preset 0.5.")
    parser.add_argument("--reanalyze-interval", type=int, default=None,
                        help="Override config.reanalyze_interval (training steps between "
                             "reanalyze calls; 0 = disabled). Preset 1024.")
    parser.add_argument("--material-value-weight", type=float, default=None,
                        help="Override config.material_value_weight (KataGo c_score "
                             "analogue). w in [0,1]: blends a material-margin WDL into the "
                             "self-play value target so two won positions are rank-able "
                             "(within-position resolution). chess only; 0 = off.")
    parser.add_argument("--material-value-scale", type=float, default=None,
                        help="Override config.material_value_scale (pawns mapping to ~tanh "
                             "saturation; 5 ≈ a rook). Preset 5.0.")
    parser.add_argument("--material-value-weight-final", type=float, default=None,
                        help="Override config.material_value_weight_final (annealing floor "
                             "the material blend weight decays toward). Preset 0.0.")
    parser.add_argument("--material-value-anneal-frac", type=float, default=None,
                        help="Override config.material_value_anneal_frac. Fraction of training "
                             "over which the material blend weight linearly decays init→final. "
                             "0 = no annealing (constant weight). E.g. 0.6 = faded by 60%% steps. "
                             "Shared timeline for the material-margin head loss weight too.")
    parser.add_argument("--use-material-head", action="store_true", default=False,
                        help="Enable the auxiliary material-margin head (KataGo score-dist "
                             "analogue): predicts current STM material from the latent state to "
                             "regularize the world model. chess only.")
    parser.add_argument("--material-head-loss-weight", type=float, default=None,
                        help="Override config.material_head_loss_weight (INITIAL aux CE weight; "
                             "annealed on --material-value-anneal-frac). Preset 0.25.")
    parser.add_argument("--material-head-loss-weight-final", type=float, default=None,
                        help="Override config.material_head_loss_weight_final (anneal floor). Preset 0.0.")
    parser.add_argument("--root-terminal-draws", action="store_true", default=False,
                        help="Enable the root repetition-draw penalty (terminal-aware search): "
                             "pin a move that completes a 2nd/3rd repetition to draw_score in MCTS "
                             "so the WINNING side avoids shuffling into a draw (side-aware; the "
                             "losing side keeps the draw). chess / GPU-resident self-play only.")
    parser.add_argument("--root-terminal-draws-min-repeats", type=int, default=None,
                        help="Override config.root_terminal_draws_min_repeats (2 = avoid any "
                             "repeat; 3 = only block completing a threefold). Preset 2.")
    parser.add_argument("--tb-root-probe", action="store_true", default=False,
                        help="Enable root Syzygy tablebase probing: classify the root's legal "
                             "moves vs tablebases (<= tb-max-pieces) and steer MCTS toward the "
                             "DTZ-optimal conversion move. chess / GPU-resident self-play only.")
    parser.add_argument("--tb-path", type=str, default=None,
                        help="Directory of Syzygy tablebase files (override config.tb_path).")
    parser.add_argument("--tb-max-pieces", type=int, default=None,
                        help="Max total pieces to probe (override config.tb_max_pieces, preset 5).")
    parser.add_argument("--tb-dtz-weight", type=float, default=None,
                        help="DTZ ranking weight among winning moves (override config.tb_dtz_weight, "
                             "preset 0.05; 0 = flat win value).")
    parser.add_argument("--tb-value-weight", type=float, default=None,
                        help="Tablebase VALUE relabeling weight (Lc0 rescorer analogue): blend the "
                             "DTZ-shaped Syzygy position value into the VALUE target at TB plies "
                             "(override config.tb_value_weight, preset 0.0=off; 1.0=full replace). "
                             "Keep selfplay-q-ratio≈0. WDL value head only.")
    parser.add_argument("--tb-value-hard", action="store_true", default=False,
                        help="Lc0-style HARD WDL value target at TB plies: one-hot win/draw/loss "
                             "instead of soft eval_to_wdl (which caps W-L ~0.88). Saturates Q near ±1, "
                             "crisp win/draw separation, activates the |Q|-gated MLH. Pair with "
                             "--tb-value-dtz-shape 0.0. Warmstart stays soft.")
    parser.add_argument("--tb-value-dtz-shape", type=float, default=None,
                        help="Shaping of the per-position TB value (override config.tb_value_dtz_shape, "
                             "preset 0.5): 0=flat WDL win=+1; >0 ranks wins by own DTZ "
                             "(closer mate→closer +1, floored at 1-shape).")
    parser.add_argument("--tb-moves-left-weight", type=float, default=None,
                        help="Moves-left DTM relabel weight (Lc0 Gaviota MLH rescoring): replace the "
                             "moves-left target at in-TB decisive plies with |DTM| (override "
                             "config.tb_moves_left_weight, preset 0.0=off; 1.0=full replace). "
                             "Needs --tb-gaviota-path + a moves-left head.")
    parser.add_argument("--tb-gaviota-path", type=str, default=None,
                        help="Directory of Gaviota .gtb.cp4 DTM tablebases (override config.tb_gaviota_path).")
    parser.add_argument("--ml-slope", type=float, default=None,
                        help="Moves-left MCTS utility slope: ml_term = sign(-Q)·clamp(ml_slope·child_m, "
                             "max=ml_max_effect) (override config.ml_slope, preset 0.005). Larger = stronger "
                             "per-move distance-to-mate steering. Raise ml_max_effect with it or it clips.")
    parser.add_argument("--ml-max-effect", type=float, default=None,
                        help="Cap on the moves-left MCTS utility magnitude (override config.ml_max_effect, "
                             "preset 0.1). Lc0 production uses 0.0345 (a small tiebreak).")
    parser.add_argument("--ml-threshold", type=float, default=None,
                        help="|Q| above which the moves-left utility engages (override config.ml_threshold, "
                             "preset 0.3). Lc0 uses 0.8 so the speed bonus only nudges among already-winning "
                             "moves and never trades away the win.")
    parser.add_argument("--moves-left-head-planes", type=int, default=None,
                        help="Moves-left head input projection width (override config.moves_left_head_planes, "
                             "preset 1). >1 widens the 1×1 bottleneck so the head can extract DTM from the "
                             "latent (latent has DTZ at corr 0.80 but the 1-plane head reads 0.035).")
    parser.add_argument("--moves-left-head-blocks", type=int, default=None,
                        help="Pre-projection residual blocks on the moves-left head (override "
                             "config.moves_left_head_blocks, preset 0).")
    parser.add_argument("--value-head-planes", type=int, default=None,
                        help="Value head input projection width (override config.value_head_planes, preset 1).")
    parser.add_argument("--grad-checkpoint-attention", action="store_true", default=False,
                        help="Recompute attention layers in backward (exact math, ~25-30% slower "
                             "train steps, large activation-memory savings). Needed for "
                             "chess_hybrid_xl batch-512 on 32GB cards.")
    parser.add_argument("--batch-mixture-schedule", type=str, default=None,
                        help='JSON schedule of declarative batch composition (task #19), e.g. '
                             '\'[[0.0,{"warmstart":0.4,"anchor":0.2,"selfplay":0.4}],'
                             '[0.6,{"warmstart":0.1,"anchor":0.1,"selfplay":0.8}]]\'. '
                             "Supersedes warmstart-sample-frac stratification.")
    parser.add_argument("--anchor-max-size", type=int, default=None,
                        help="Cap the TB-anchor pool in the main buffer (three-pool eviction). "
                             "Unset = legacy emergent anchor volume.")
    parser.add_argument("--position-sampling", choices=["per_game", "per_ply"], default=None,
                        help="Position sampling within games: per_game (legacy) or per_ply "
                             "(every stored ply equally likely; removes short-game overweighting).")
    parser.add_argument("--reward-head-planes", type=int, default=None,
                        help="Width of the dynamics-reward head's 1x1 projection (config default 1). "
                             "The reward head is the search's in-tree mate detector; 8 matches the "
                             "value/moves-left heads. Changes parameter shapes (fresh run or "
                             "--warmstart-body only).")
    parser.add_argument("--symmetry-augment", action="store_true", default=False,
                        help="D4 symmetry augmentation on pawnless castle-free training windows: "
                             "random dihedral transform per sample (obs+actions+policies+masks), "
                             "forcing relative-geometry features over absolute-square memorization.")
    parser.add_argument("--seed-curriculum", action="store_true", default=False,
                        help="DTM-stratified reverse curriculum for endgame seeds: sample seeds "
                             "with |DTM| <= a cap ramping dtm_easy->dtm_hard over "
                             "seed_curriculum_anneal_frac of training (short mating chains first). "
                             "Needs <archive>.dtm from scripts/annotate_seed_dtm.py.")
    parser.add_argument("--merged-seed-batch", action="store_true", default=False,
                        help="Run normal + seeded self-play games in ONE resident sweep "
                             "(mixed start states, per-game opening masks) instead of "
                             "sequential sub-batches — one straggler tail, ~15-30% self-play "
                             "wall-clock saving.")
    parser.add_argument("--opening-mix-mean-plies", type=int, default=None,
                        help="Opening diversity ε-mixture: normal games open with r ~ U[1, 2·mean] "
                             "searchless plies — uniform-random w.p. --opening-uniform-frac "
                             "(model-independent diversity floor), else raw-policy softmax at "
                             "--opening-policy-temp (KataGo initGamesWithPolicy). Opening plies "
                             "store ZERO policy targets. 0/unset = off.")
    parser.add_argument("--opening-policy-temp", type=float, default=None,
                        help="Softmax temperature for policy-sampled opening plies (config 1.5).")
    parser.add_argument("--opening-uniform-frac", type=float, default=None,
                        help="Fraction of games opening with uniform-random plies (config 0.15).")
    parser.add_argument("--per-alpha", type=float, default=None,
                        help="PER priority exponent (override config.per_alpha). 0 = uniform "
                             "sampling — the MuZero-board-games/KataGo/Lc0-consistent choice; "
                             "also prevents value-TD priorities from starving anchor games "
                             "whose easy value targets hide unlearned policy content.")
    parser.add_argument("--resign-exempt-seeded", action="store_true", default=False,
                        help="Exempt seeded (start_fen) games from resignation: their value labels "
                             "are TB-true regardless, and resignation truncates exactly the "
                             "conversion-practice tails seeding exists to generate. Makes "
                             "seed/conversion and seed/mate_rate honest skill metrics.")
    parser.add_argument("--no-attention", action="store_true", default=False,
                        help="Disable the attention backbone (use the conv residual tower) even "
                             "when the game preset enables it — for conv-vs-attention A/Bs and "
                             "resuming conv-era checkpoints.")
    parser.add_argument("--tb-anchor-path", type=str, default=None,
                        help="Directory of TB anchor shards (scripts/gen_tb_anchor_games.py): "
                             "tablebase-optimal demonstration games injected into the rolling "
                             "buffer on an interval, cycling forever (persistent endgame anchor).")
    parser.add_argument("--tb-anchor-games", type=int, default=64,
                        help="TB anchor games injected per interval (default 64).")
    parser.add_argument("--tb-anchor-interval", type=int, default=256,
                        help="Training steps between TB anchor injections (default 256).")
    parser.add_argument("--tb-rollout-fill", action="store_true", default=False,
                        help="Win adjudication by demonstration: truncate a non-seeded self-play "
                             "game at its first decisive in-TB ply when the played outcome "
                             "contradicts the TB verdict, and finish it with TB-optimal play by "
                             "both sides (true decisive z for the whole trajectory + an "
                             "on-distribution conversion demonstration). Needs --tb-root-probe.")
    parser.add_argument("--tb-steer-policy", action="store_true", default=False,
                        help="Restore the search-side DTZ value bias (policy steering) in _select. "
                             "OFF by default — Lc0-faithful: inject TB signal via relabels, not search.")
    parser.add_argument("--tb-policy-weight", type=float, default=None,
                        help="Soft TB POLICY relabel weight (Lc0 DTZ policy boost, safe sans KLDGain): "
                             "blend a win-preserving distribution into the policy TARGET at TB plies, "
                             "(1-w)*visits + w*tb (override config.tb_policy_weight, preset 0=off). Needs --tb-root-probe.")
    parser.add_argument("--tb-policy-weight-final", type=float, default=None,
                        help="Final TB policy-relabel weight after annealing (override config.tb_policy_weight_final).")
    parser.add_argument("--tb-policy-anneal-frac", type=float, default=None,
                        help="Fraction of training over which tb_policy_weight decays to its final "
                             "(override config.tb_policy_anneal_frac; 0=constant). Fade the teacher to avoid crutch.")
    parser.add_argument("--tb-policy-temp", type=float, default=None,
                        help="Softmax temperature of the relabeled policy TARGET over win-preserving moves "
                             "(override config.tb_policy_temp, preset 0.3). Lower = sharper (more mass on the "
                             "DTZ-best winning move). NOT the MCTS search temperature.")
    parser.add_argument("--tb-relabel-workers", type=int, default=None,
                        help="Deferred TB relabel pool size (override config.tb_relabel_workers, preset 0). "
                             "When steering is off the value/DTM/policy targets run in one batched post-game "
                             "pass; >1 fans the probes across this many spawn workers (each opens its own "
                             "tablebases). 0/1 = single-process deferred pass. Removes the prober from the hot path.")
    parser.add_argument("--endgame-seed-frac", type=float, default=None,
                        help="Fraction of each self-play round seeded from tablebase endgame FENs "
                             "(on-policy curriculum; override config.endgame_seed_frac, preset 0=off).")
    parser.add_argument("--endgame-seed-archive", type=str, default=None,
                        help="Path to the endgame-seed FEN list (scripts/generate_endgame_seeds.py output .txt).")
    parser.add_argument("--prefetch-batches", action="store_true", default=False,
                        help="Overlap sample_batch (CPU) with the GPU train step via a background "
                             "prefetch thread (~halves training-phase step time). Off by default.")
    parser.add_argument("--resign-enabled", action="store_true", default=False,
                        help="Enable post-hoc consecutive-move resignation: if STM root "
                             "value < --resign-threshold for --resign-consecutive own-moves, "
                             "truncate + relabel the game as a decisive loss (label protection).")
    parser.add_argument("--resign-holdout-frac", type=float, default=None,
                        help="Override config.resign_holdout_frac (AlphaZero-style): fraction of "
                             "would-resign games played to completion to measure the false-positive "
                             "rate (self_play/resign_false_positive_rate; tune threshold to <5%%). Preset 0.15.")
    parser.add_argument("--resign-threshold", type=float, default=None,
                        help="Override config.resign_threshold (STM-POV root value; ≈≤5%% "
                             "expected score). Preset -0.9.")
    parser.add_argument("--resign-consecutive", type=int, default=None,
                        help="Override config.resign_consecutive (consecutive own-moves below "
                             "threshold to trigger resignation). Preset 5.")
    parser.add_argument("--mask-illegal-policy", action="store_true", default=False,
                        help="Enable legal-move policy masking (config.mask_illegal_policy). "
                             "Keeps the standard full-softmax policy CE and ADDS a penalty "
                             "on the full-softmax probability mass landing on illegal moves, "
                             "driving it below the CE's natural floor (~17% at 22k for chess). "
                             "Recommended for chess; default off.")
    parser.add_argument("--illegal-policy-penalty", type=float, default=None,
                        help="Override config.illegal_policy_penalty (weight on the "
                             "illegal-mass penalty; only active with --mask-illegal-policy).")
    parser.add_argument("--decisive-retention-multiplier", type=float, default=None,
                        help="Override config.decisive_retention_multiplier (M). >1 keeps "
                             "DECISIVE games in the replay buffer ~M× longer than draws "
                             "(retention-weighted eviction), raising decisive density. "
                             "1.0 = FIFO. At ~5%% decisive inflow: M=7 → ~27%% of buffer decisive.")
    parser.add_argument("--policy-head-type", choices=["flat", "conv", "from_to"], default=None,
                        help="Override config.policy_head_type. 'conv' = AlphaZero spatial "
                             "73-plane policy head (chess only). 'from_to' = relational bilinear "
                             "from/to-square head (2026-07-08 arch sweep arm C: +48%% proxy MCTS "
                             "conversion vs conv at matched steps; codec-parity tested). Changes "
                             "the policy-head shape, so a checkpoint trained with another type "
                             "won't load its policy head.")
    parser.add_argument("--replay-buffer-size", type=int, default=None,
                        help="Override config.replay_buffer_size (total games held "
                             "in the replay buffer). Scale with games/round to keep "
                             "passes-per-position near target (~3): 5120 @512 games, "
                             "10240 @1024.")
    parser.add_argument("--warmstart-buffer-size", type=int, default=None,
                        help="Override config.warmstart_buffer_size. Enables the "
                             "TWO-POOL buffer: this many slots are reserved for "
                             "warmstart (Stockfish) games and evict FIFO only "
                             "within that pool, so self-play traffic can NEVER "
                             "displace the warmstart anchor. Required for "
                             "--warmstart-sample-frac > 0 to keep working past "
                             "the point where self-play would otherwise drain the "
                             "warmstart games (the single-pool drain that caused "
                             "the value-head collapse at step ~15.8k). Must be "
                             "< replay_buffer_size; the remainder is the self-play pool.")
    parser.add_argument("--use-gpu-chess", action="store_true",
                        help="Use the GPU-resident chess env (GpuChessGame) for self-play.")
    parser.add_argument("--use-tensor-mcts", action="store_true",
                        help="Use the GPU tensor-native MCTS (TensorMCTS) instead of BatchedMCTS.")
    parser.add_argument("--use-gpu-resident-self-play", action="store_true",
                        help="Use the fully GPU-resident self-play loop (0 syncs/ply). "
                             "Requires --use-tensor-mcts and --use-gpu-chess.")
    parser.add_argument("--tensor-mcts-select-backend", default=None,
                        choices=["compile", "triton", "eager"],
                        help="MCTS PUCT backend. 'triton' is fastest (~1.2× over compile).")
    parser.add_argument("--tensor-mcts-subtree-reuse", action="store_true",
                        help="Enable subtree reuse across plies (carry chosen child's "
                             "subtree into next ply's search; doubles tree storage).")
    parser.add_argument("--tensor-mcts-hidden-dtype", default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Storage dtype for MCTS node_hidden. fp16 halves memory.")
    parser.add_argument("--tensor-mcts-compile-net", action="store_true",
                        help="torch.compile the per-sim network forward inside MCTS "
                             "(recurrent_inference etc.). ~1.4x faster self-play search; "
                             "exact net-output parity. See config.tensor_mcts_compile_net.")
    parser.add_argument("--max-plies", type=int, default=None,
                        help="Override config.max_plies (self-play ply cap; draw if reached). "
                             "Bounds GPU-resident self-play memory (per-ply stacks grow with the "
                             "longest game) and wall-clock. Lower for long-shuffling cold runs.")
    parser.add_argument("--no-consistency-loss", action="store_true",
                        help="Disable EfficientZero SimSiam consistency loss for this run "
                             "(falsifier for the action-blind dynamics collapse hypothesis).")
    parser.add_argument("--consistency-single-frame-target", dest="consistency_single_frame_target",
                        action="store_true", default=None,
                        help="Use single-frame (newest-ply, zero-padded) consistency target "
                             "instead of the full T-frame stack. De-trivializes the SimSiam target.")
    parser.add_argument("--no-consistency-single-frame-target", dest="consistency_single_frame_target",
                        action="store_false",
                        help="Force the legacy full-stack consistency target.")
    parser.add_argument("--use-inverse-dynamics-loss", dest="use_inverse_dynamics_loss",
                        action="store_true", default=None,
                        help="Enable the ICM inverse-dynamics aux loss (predict a_k from "
                             "h_k,h_{k+1}). Validated fix for action-blind dynamics.")
    parser.add_argument("--no-inverse-dynamics-loss", dest="use_inverse_dynamics_loss",
                        action="store_false",
                        help="Disable the inverse-dynamics aux loss (e.g. to A/B against it).")
    parser.add_argument("--inverse-dynamics-loss-weight", type=float, default=None,
                        help="Weight on the inverse-dynamics aux loss. Overrides config.")
    parser.add_argument("--value-head-init-std", type=float, default=None,
                        help="Std for small-normal init of the value head's last linear "
                             "(0.0 = zero-init default). >0 lets body gradient flow at cold start.")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    config = get_config(args.game)
    config.device = device
    if args.steps is not None:
        config.training_steps = args.steps
    if args.sample_k is not None:
        config.sample_k = args.sample_k
    if args.use_gumbel:
        config.use_gumbel = True
    if args.gumbel_m is not None:
        config.gumbel_num_considered = args.gumbel_m
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.root_heavy_loss:
        config.use_root_heavy_loss = True
    if args.max_buf_save_games is not None:
        config.max_buf_save_games = args.max_buf_save_games
    if args.stockfish_injection_games is not None:
        config.stockfish_injection_games = args.stockfish_injection_games
    if args.stockfish_injection_interval is not None:
        config.stockfish_injection_interval = args.stockfish_injection_interval
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
    if args.num_parallel_games is not None:
        config.num_parallel_games = args.num_parallel_games
    if args.num_self_play_games is not None:
        config.num_self_play_games = args.num_self_play_games
    if args.self_play_interval is not None:
        config.self_play_interval = args.self_play_interval
    if args.warmstart_sample_frac is not None:
        config.warmstart_sample_frac = args.warmstart_sample_frac
    if args.warmstart_sample_frac_final is not None:
        config.warmstart_sample_frac_final = args.warmstart_sample_frac_final
    if args.warmstart_anneal_frac is not None:
        config.warmstart_anneal_frac = args.warmstart_anneal_frac
    if args.self_play_warmup_steps is not None:
        config.self_play_warmup_steps = args.self_play_warmup_steps
    # Fractional form takes precedence and resolves against the FINAL training_steps
    # (set above from --steps), so it stays aligned with the mixture schedule.
    if args.self_play_warmup_frac is not None:
        config.self_play_warmup_frac = args.self_play_warmup_frac
        config.self_play_warmup_steps = int(round(args.self_play_warmup_frac
                                                   * config.training_steps))
        print(f"self-play warmup: frac {args.self_play_warmup_frac:.4f} x "
              f"{config.training_steps} steps = {config.self_play_warmup_steps} "
              f"(fractional; matches --batch-mixture-schedule notation)")
    if args.warmstart_q_ratio is not None:
        config.warmstart_q_ratio = args.warmstart_q_ratio
    if args.selfplay_q_ratio is not None:
        config.selfplay_q_ratio = args.selfplay_q_ratio
    if args.temperature_drop_step is not None:
        config.temperature_drop_step = args.temperature_drop_step
    if args.dirichlet_alpha is not None:
        config.dirichlet_alpha = args.dirichlet_alpha
    if args.use_moves_left is not None:
        config.use_moves_left = args.use_moves_left
        if not args.use_moves_left:
            config.moves_left_mcts = False  # disabling the head also disables its search utility
    if args.repetition_penalty is not None:
        config.repetition_penalty = args.repetition_penalty
    if args.repetition_penalty_window is not None:
        config.repetition_penalty_window = args.repetition_penalty_window
    if args.repetition_penalty_decay is not None:
        config.repetition_penalty_decay = args.repetition_penalty_decay
    if args.draw_score is not None:
        config.draw_score = args.draw_score
    if args.eval_to_wdl_alpha is not None:
        config.eval_to_wdl_alpha = args.eval_to_wdl_alpha
    if args.eval_to_wdl_beta is not None:
        config.eval_to_wdl_beta = args.eval_to_wdl_beta
    if args.decisive_sample_frac is not None:
        config.decisive_sample_frac = args.decisive_sample_frac
    if args.reanalyze_interval is not None:
        config.reanalyze_interval = args.reanalyze_interval
    if args.material_value_weight is not None:
        config.material_value_weight = args.material_value_weight
    if args.material_value_scale is not None:
        config.material_value_scale = args.material_value_scale
    if args.material_value_weight_final is not None:
        config.material_value_weight_final = args.material_value_weight_final
    if args.material_value_anneal_frac is not None:
        config.material_value_anneal_frac = args.material_value_anneal_frac
    if args.use_material_head:
        config.use_material_head = True
    if args.material_head_loss_weight is not None:
        config.material_head_loss_weight = args.material_head_loss_weight
    if args.material_head_loss_weight_final is not None:
        config.material_head_loss_weight_final = args.material_head_loss_weight_final
    if args.root_terminal_draws:
        config.root_terminal_draws = True
    if args.root_terminal_draws_min_repeats is not None:
        config.root_terminal_draws_min_repeats = args.root_terminal_draws_min_repeats
    if args.tb_root_probe:
        config.tb_root_probe = True
    if args.tb_path is not None:
        config.tb_path = args.tb_path
    if args.tb_max_pieces is not None:
        config.tb_max_pieces = args.tb_max_pieces
    if args.tb_dtz_weight is not None:
        config.tb_dtz_weight = args.tb_dtz_weight
    if args.tb_value_weight is not None:
        config.tb_value_weight = args.tb_value_weight
    if args.tb_value_hard:
        config.tb_value_hard = True
    if args.tb_value_dtz_shape is not None:
        config.tb_value_dtz_shape = args.tb_value_dtz_shape
    if args.tb_moves_left_weight is not None:
        config.tb_moves_left_weight = args.tb_moves_left_weight
    if args.tb_gaviota_path is not None:
        config.tb_gaviota_path = args.tb_gaviota_path
    if args.ml_slope is not None:
        config.ml_slope = args.ml_slope
    if args.ml_max_effect is not None:
        config.ml_max_effect = args.ml_max_effect
    if args.ml_threshold is not None:
        config.ml_threshold = args.ml_threshold
    if args.grad_checkpoint_attention:
        config.grad_checkpoint_attention = True
    if args.batch_mixture_schedule is not None:
        import json
        sched = json.loads(args.batch_mixture_schedule)
        config.batch_mixture_schedule = [(float(f), dict(m)) for f, m in sched]
    if args.anchor_max_size is not None:
        config.anchor_max_size = args.anchor_max_size
    if args.position_sampling is not None:
        config.position_sampling = args.position_sampling
    if args.reward_head_planes is not None:
        config.reward_head_planes = args.reward_head_planes
    if args.moves_left_head_planes is not None:
        config.moves_left_head_planes = args.moves_left_head_planes
    if args.moves_left_head_blocks is not None:
        config.moves_left_head_blocks = args.moves_left_head_blocks
    if args.value_head_planes is not None:
        config.value_head_planes = args.value_head_planes
    if args.symmetry_augment:
        config.symmetry_augment = True
    if args.seed_curriculum:
        config.seed_curriculum = True
    if args.merged_seed_batch:
        config.merged_seed_batch = True
    if args.opening_mix_mean_plies is not None:
        config.opening_mix_mean_plies = args.opening_mix_mean_plies
    if args.opening_policy_temp is not None:
        config.opening_policy_temp = args.opening_policy_temp
    if args.opening_uniform_frac is not None:
        config.opening_uniform_frac = args.opening_uniform_frac
    if args.per_alpha is not None:
        config.per_alpha = args.per_alpha
    if args.resign_exempt_seeded:
        config.resign_exempt_seeded = True
    if args.no_attention:
        config.use_repr_attention = False
        config.use_dyn_attention = False
        config.use_pred_attention = False
    if args.tb_anchor_path is not None:
        config.tb_anchor_path = args.tb_anchor_path
        config.tb_anchor_games = args.tb_anchor_games
        config.tb_anchor_interval = args.tb_anchor_interval
    if args.tb_rollout_fill:
        config.tb_rollout_fill = True
    if args.tb_steer_policy:
        config.tb_steer_policy = True
    if args.tb_policy_weight is not None:
        config.tb_policy_weight = args.tb_policy_weight
    if args.tb_policy_weight_final is not None:
        config.tb_policy_weight_final = args.tb_policy_weight_final
    if args.tb_policy_anneal_frac is not None:
        config.tb_policy_anneal_frac = args.tb_policy_anneal_frac
    if args.tb_policy_temp is not None:
        config.tb_policy_temp = args.tb_policy_temp
    if args.tb_relabel_workers is not None:
        config.tb_relabel_workers = args.tb_relabel_workers
    if args.endgame_seed_frac is not None:
        config.endgame_seed_frac = args.endgame_seed_frac
    if args.endgame_seed_archive is not None:
        config.endgame_seed_archive = args.endgame_seed_archive
    if args.prefetch_batches:
        config.prefetch_batches = True
    if args.resign_enabled:
        config.resign_enabled = True
    if args.resign_threshold is not None:
        config.resign_threshold = args.resign_threshold
    if args.resign_consecutive is not None:
        config.resign_consecutive = args.resign_consecutive
    if args.resign_holdout_frac is not None:
        config.resign_holdout_frac = args.resign_holdout_frac
    if args.decisive_retention_multiplier is not None:
        config.decisive_retention_multiplier = args.decisive_retention_multiplier
    if args.policy_head_type is not None:
        config.policy_head_type = args.policy_head_type
    if args.mask_illegal_policy:
        config.mask_illegal_policy = True
    if args.illegal_policy_penalty is not None:
        config.illegal_policy_penalty = args.illegal_policy_penalty
    if args.replay_buffer_size is not None:
        config.replay_buffer_size = args.replay_buffer_size
    if args.warmstart_buffer_size is not None:
        if args.warmstart_buffer_size >= config.replay_buffer_size:
            parser.error(
                f"--warmstart-buffer-size ({args.warmstart_buffer_size}) must be "
                f"< replay_buffer_size ({config.replay_buffer_size}); the remainder "
                f"is the self-play pool.")
        config.warmstart_buffer_size = args.warmstart_buffer_size
    if args.use_gpu_chess:
        config.use_gpu_chess = True
    if args.use_tensor_mcts:
        config.use_tensor_mcts = True
    if args.use_gpu_resident_self_play:
        config.use_gpu_resident_self_play = True
    if args.tensor_mcts_select_backend is not None:
        config.tensor_mcts_select_backend = args.tensor_mcts_select_backend
    if args.tensor_mcts_subtree_reuse:
        config.tensor_mcts_subtree_reuse = True
    if args.tensor_mcts_hidden_dtype is not None:
        config.tensor_mcts_hidden_dtype = args.tensor_mcts_hidden_dtype
    if args.tensor_mcts_compile_net:
        config.tensor_mcts_compile_net = True
    if args.max_plies is not None:
        config.max_plies = args.max_plies
    if args.no_consistency_loss:
        config.use_consistency_loss = False
    if args.consistency_single_frame_target is not None:
        config.consistency_single_frame_target = args.consistency_single_frame_target
    if args.use_inverse_dynamics_loss is not None:
        config.use_inverse_dynamics_loss = args.use_inverse_dynamics_loss
    if args.inverse_dynamics_loss_weight is not None:
        config.inverse_dynamics_loss_weight = args.inverse_dynamics_loss_weight
    if args.value_head_init_std is not None:
        config.value_head_init_std = args.value_head_init_std

    # Use CPU AMP settings appropriately
    if device == "cpu":
        config.use_amp = False

    game = get_game(args.game)

    # Run dirs are keyed by config.game (the ENGINE name, e.g. "chess"), not the
    # preset name passed as --game (e.g. "chess_hybrid") — the trainer's writer
    # and checkpointer use config.game, so the banner must too or it prints
    # paths that don't exist (chess_hybrid/... vs the real chess/...).
    game_dir = config.game
    run_id = args.run_id or generate_run_id(
        Path(args.checkpoints_dir) / game_dir,
        Path(args.log_dir) / game_dir,
    )
    print(f"Run ID: {run_id}")
    print(f"  Checkpoints: {Path(args.checkpoints_dir) / game_dir / run_id}")
    print(f"  TensorBoard: {Path(args.log_dir) / game_dir / run_id}")

    network = MuZeroNetwork(
        observation_channels=game.num_planes * getattr(config, "history_frames", 1),
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h,
        latent_w=config.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        reward_head_planes=getattr(config, "reward_head_planes", 1),
        action_embed_dim=getattr(config, "action_embed_dim", 16),
        use_consistency_loss=config.use_consistency_loss,
        proj_hid=config.proj_hid,
        proj_out=config.proj_out,
        pred_hid=config.pred_hid,
        pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
        value_head_type=getattr(config, "value_head_type", "support"),
        draw_score=getattr(config, "draw_score", 0.0),
        value_head_init_std=getattr(config, "value_head_init_std", 0.0),
        use_inverse_dynamics_loss=getattr(config, "use_inverse_dynamics_loss", False),
        inverse_dynamics_hidden=getattr(config, "inverse_dynamics_hidden", 256),
        policy_head_type=getattr(config, "policy_head_type", "flat"),
        use_moves_left=getattr(config, "use_moves_left", False),
        moves_left_support_size=getattr(config, "moves_left_support_size", 10),
        use_material_head=getattr(config, "use_material_head", False),
        material_head_support_size=getattr(config, "material_head_support_size", 8),
        value_head_planes=getattr(config, "value_head_planes", 1),
        value_head_blocks=getattr(config, "value_head_blocks", 0),
        moves_left_head_planes=getattr(config, "moves_left_head_planes", 1),
        moves_left_head_blocks=getattr(config, "moves_left_head_blocks", 0),
        use_repr_attention=getattr(config, "use_repr_attention", False),
        use_dyn_attention=getattr(config, "use_dyn_attention", False),
        use_pred_attention=getattr(config, "use_pred_attention", False),
        use_smolgen=getattr(config, "use_smolgen", True),
        attn_layers=getattr(config, "attn_layers", 4),
        attn_heads=getattr(config, "attn_heads", 4),
        pred_attn_layers=getattr(config, "pred_attn_layers", 2),
        hybrid_stem_blocks=getattr(config, "hybrid_stem_blocks", 0),
    )
    if getattr(config, "grad_checkpoint_attention", False):
        from src.model.attention import BoardAttentionEncoder
        n_ck = 0
        for m in network.modules():
            if isinstance(m, BoardAttentionEncoder):
                m.grad_checkpoint = True
                n_ck += 1
        print(f"Activation checkpointing ON for {n_ck} attention encoder(s)")

    trainer = MuZeroTrainer(
        config, game, network, run_id,
        device=device,
        log_dir=args.log_dir,
        checkpoints_dir=args.checkpoints_dir,
    )

    if args.warmstart_body:
        trainer.load_body_warmstart(args.warmstart_body)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        if args.reset_injection_cursor:
            old_cursor = trainer._injection_loaded
            trainer._injection_loaded = 0
            print(f"Reset Stockfish injection cursor: {old_cursor} → 0 "
                  f"(--reset-injection-cursor)")

    if args.stockfish_injection_path:
        import glob
        import random
        p = args.stockfish_injection_path
        if os.path.isdir(p):
            # Recursive glob — parallel generation lays shards under
            # bucket_*/worker_*/subdirs.
            injection_paths = sorted(glob.glob(os.path.join(p, "**", "*.pkl"),
                                                recursive=True))
        else:
            injection_paths = sorted(glob.glob(p))
        if not injection_paths:
            raise FileNotFoundError(
                f"No .pkl shards found under --stockfish-injection-path: {p}"
            )

        # Deterministic shuffle so multi-bucket pools (e.g. asymmetric-teacher
        # 8v5/8v6/8v7/8v8 layout) mix evenly through the buffer instead of
        # feeding one bucket at a time in curriculum order (which biases the
        # position distribution the model sees across warmstart). Seed=0 gives
        # reproducible ordering across resumes; negative seed preserves the
        # legacy sorted-alphabetical order.
        seed = args.stockfish_injection_shuffle_seed
        if seed >= 0:
            random.Random(seed).shuffle(injection_paths)

        trainer.set_injection_shards(injection_paths)
        shuffle_note = (f"shuffled with seed={seed}" if seed >= 0
                        else "sorted-alphabetical (shuffle disabled)")
        print(f"Stockfish injection: {len(injection_paths)} shard(s) attached from {p} "
              f"[{shuffle_note}] (cursor resumed at {trainer._injection_loaded} games consumed)")

    trainer.train()


if __name__ == "__main__":
    main()
