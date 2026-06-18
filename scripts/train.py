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
    if name in ("chess", "chess_small"):
        from src.games.chess import ChessGame
        return ChessGame()
    if name == "checkers":
        from src.games.checkers import Checkers
        return Checkers()
    raise ValueError(f"Unknown game: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train MuZero")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "connect4", "chess", "chess_small", "checkers"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: auto-generate YYYY_MM_DD_NNNN). "
                             "Pass an existing ID to continue writing into that run's dirs.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
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
    parser.add_argument("--warmstart-sample-frac", type=float, default=None,
                        help="Override config.warmstart_sample_frac. Set to 0.0 "
                             "for a pure-self-play run with no warmstart anchor "
                             "(also pass --stockfish-injection-games 0 + "
                             "--stockfish-injection-interval 0 to disable injection).")
    parser.add_argument("--self-play-warmup-steps", type=int, default=None,
                        help="Override config.self_play_warmup_steps. Two-phase "
                             "(Option A) curriculum: self-play + reanalyze are gated "
                             "OFF for this many steps (pure supervised Stockfish "
                             "warmstart pretrain), then flip on for the rest of "
                             "training. The warmstart anchor persists into phase 2. "
                             "0 = legacy pool-exhaustion gate.")
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
    parser.add_argument("--policy-head-type", choices=["flat", "conv"], default=None,
                        help="Override config.policy_head_type. 'conv' = AlphaZero spatial "
                             "73-plane policy head (chess only; removes the flat head's "
                             "channel bottleneck). Changes the policy-head shape, so a "
                             "checkpoint trained with the other type won't load its policy head.")
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
    if args.warmstart_sample_frac is not None:
        config.warmstart_sample_frac = args.warmstart_sample_frac
    if args.self_play_warmup_steps is not None:
        config.self_play_warmup_steps = args.self_play_warmup_steps
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
    if args.decisive_retention_multiplier is not None:
        config.decisive_retention_multiplier = args.decisive_retention_multiplier
    if args.policy_head_type is not None:
        config.policy_head_type = args.policy_head_type
    if args.mask_illegal_policy:
        config.mask_illegal_policy = True
    if args.illegal_policy_penalty is not None:
        config.illegal_policy_penalty = args.illegal_policy_penalty
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

    run_id = args.run_id or generate_run_id(
        Path(args.checkpoints_dir) / args.game,
        Path(args.log_dir) / args.game,
    )
    print(f"Run ID: {run_id}")
    print(f"  Checkpoints: {Path(args.checkpoints_dir) / args.game / run_id}")
    print(f"  TensorBoard: {Path(args.log_dir) / args.game / run_id}")

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
    )

    trainer = MuZeroTrainer(
        config, game, network, run_id,
        device=device,
        log_dir=args.log_dir,
        checkpoints_dir=args.checkpoints_dir,
    )

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
