"""Sibling-ranking probe — reconfirm finding #2.

Claim under test: the value head is calibrated ACROSS positions (corr ~0.85
with Stockfish) but cannot rank the SIBLING moves WITHIN a single position
(near-zero rank-correlation among the children of one node) -> MCTS gets no
useful value signal to allocate visits among candidate moves.

For each on-distribution position (model rollout: random opening then MCTS):
  - model per-move value = -V(child position)  (my POV; child is opponent-to-move)
  - Stockfish per-move eval (my POV) via multipv over all legal moves
  - within-position Spearman(model, SF) over the legal moves
and as a CONTRAST, the across-position correlation of V(position) vs SF(position)
(the "calibration" number that should be ~0.85).

Run: .venv/bin/python scripts/probe_sibling_ranking.py --checkpoint <path.pt>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.engine
from scipy.stats import spearmanr

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.mcts import MCTS, select_action
from src.training.replay_buffer import stack_with_history


def _spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = get_config("chess"); cfg.device = "cpu"; cfg.num_simulations = args.sims
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"]
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size, reward_support_size=cfg.reward_support_size,
        use_consistency_loss=any(k.startswith("projection.") for k in sd),
        use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale,
        value_head_type=cfg.value_head_type, draw_score=cfg.draw_score,
        use_inverse_dynamics_loss=any(k.startswith("inverse_dynamics_head.") for k in sd),
        # checkpoint has a flat policy head; default policy_head_type="flat" loads it.
    )
    net.load_state_dict(sd); net.eval()
    mcts = MCTS(net, game, cfg, "cpu")
    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    print(f"Loaded {args.checkpoint} (step {ck.get('step','?')}); sims={args.sims}, "
          f"SF depth={args.sf_depth}\n")

    def model_value(state, frames):
        """Scalar value of `state` from its side-to-move POV."""
        obs = stack_with_history(game.to_tensor(state), frames, cfg.history_frames)
        with torch.no_grad():
            v = net.initial_inference(obs.unsqueeze(0))[2]
        return float(v.squeeze().item())

    state = game.reset(); frames = []; ply = 0; n_open = 8
    per_pos_spearman, best_agree, model_move_sf_rank = [], [], []
    pos_modelV, pos_sfV = [], []   # across-position contrast (narrow, midgame only)
    pool_model, pool_sf = [], []   # POOLED across all child positions (WIDE spread)
    collected = 0
    while collected < args.positions and ply < 400:
        legal = game.legal_actions(state)
        if len(legal) < 2:
            state = game.reset(); frames = []; ply = 0; continue
        if ply < n_open:
            action = int(np.random.choice(legal))
            frames.append(game.to_tensor(state))
            state, _, done = game.step(state, action)
            if done: state = game.reset(); frames = []; ply = 0; continue
            ply += 1; continue

        board = state.board
        if len(legal) >= 3 and not board.is_game_over():
            # across-position calibration sample
            pos_modelV.append(model_value(state, frames))
            info0 = eng.analyse(board, chess.engine.Limit(depth=args.sf_depth))
            pos_sfV.append(info0["score"].pov(board.turn).score(mate_score=10000) / 100.0)

            # per-move model value (my POV = -V(child)) and SF eval (my POV)
            sf_info = eng.analyse(board, chess.engine.Limit(depth=args.sf_depth),
                                  multipv=min(len(legal), 60))
            sf_cp = {d["pv"][0]: d["score"].pov(board.turn).score(mate_score=10000)
                     for d in sf_info}
            mv_model, mv_sf, moves = [], [], []
            child_frames = frames + [game.to_tensor(state)]
            for a in legal:
                nxt, _, _ = game.step(state, a)
                mv = nxt.board.move_stack[-1]
                if mv not in sf_cp:
                    continue
                mv_model.append(-model_value(nxt, child_frames))   # my POV
                mv_sf.append(sf_cp[mv]); moves.append(mv)
            pool_model.extend(mv_model)
            pool_sf.extend([min(max(c, -1500), 1500) for c in mv_sf])  # clip mate outliers
            if len(moves) >= 3:
                sp = _spearman(mv_model, mv_sf)
                per_pos_spearman.append(sp)
                model_best = moves[int(np.argmax(mv_model))]
                sf_best = max(sf_cp, key=sf_cp.get)
                best_agree.append(1.0 if model_best == sf_best else 0.0)
                # SF rank (1=best) of the model's value-greedy move
                sf_sorted = sorted(sf_cp, key=sf_cp.get, reverse=True)
                model_move_sf_rank.append(sf_sorted.index(model_best) + 1)
                collected += 1
                print(f"  pos {collected:2d}: legal={len(moves):2d}  "
                      f"spearman(V,SF)={sp:+.2f}  best-agree={'Y' if best_agree[-1] else 'n'}  "
                      f"model-move SF-rank={model_move_sf_rank[-1]}")

        # advance with the model (greedy) to stay on-distribution
        obs = stack_with_history(game.to_tensor(state), frames, cfg.history_frames)
        root = mcts.run(obs, legal, add_noise=False)
        action, _ = select_action(root, temperature=0)
        frames.append(game.to_tensor(state))
        state, _, done = game.step(state, action)
        if done: state = game.reset(); frames = []; ply = 0; continue
        ply += 1
    eng.quit()

    sp = np.array([s for s in per_pos_spearman if not np.isnan(s)])
    print(f"\n=== sibling ranking over {len(per_pos_spearman)} positions ===")
    print(f"  WITHIN-position Spearman(model V, Stockfish): "
          f"median {np.median(sp):+.2f}  mean {np.mean(sp):+.2f}")
    print(f"     fraction > +0.3 : {np.mean(sp > 0.3):.0%}     fraction < 0 : {np.mean(sp < 0):.0%}")
    print(f"  best-move agreement (model value-argmax == SF best): {np.mean(best_agree):.0%}")
    print(f"  SF-rank of model's value-greedy move: median {int(np.median(model_move_sf_rank))}  "
          f"mean {np.mean(model_move_sf_rank):.1f}")
    print(f"\n  CONTRAST baselines (does V track SF when there IS spread?):")
    if len(pos_modelV) >= 3:
        c = _spearman(pos_modelV, pos_sfV)
        print(f"    across-position, NARROW set (midgame only, n={len(pos_modelV)}): "
              f"Spearman {c:+.2f}  [noisy: little eval spread on-distribution]")
    if len(pool_model) >= 3:
        cp = _spearman(pool_model, pool_sf)
        rng = np.percentile(pool_sf, [5, 95])
        print(f"    POOLED over all child positions, WIDE set (n={len(pool_model)}, "
              f"SF 5-95pct {rng[0]:.0f}..{rng[1]:.0f}cp): Spearman {cp:+.2f}")
    print("\n  Interpretation: if POOLED (wide) corr is high but WITHIN-position "
          "Spearman ~0, the value ranks big differences, not sibling moves (finding #2).")


if __name__ == "__main__":
    main()
