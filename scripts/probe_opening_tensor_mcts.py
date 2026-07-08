"""Compare TENSOR MCTS (live self-play engine) visit distribution at s0 against
the numpy reference, and measure what temperature=1.0 sampling actually yields.

This reconciles the buffer observation (ply-0 entropy 0.13b => 99% e2e4 at step
30k) with the numpy probe (visit top-share 46%). If the tensor engine produces a
much sharper visit distribution, that is the mechanistic path the real games take.

Runs N parallel copies of the start position through TensorMCTS exactly as
self-play does, then samples actions at temperature 1.0 via select_action_gpu.
"""
import argparse, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.games.chess_gpu import GpuChessGame
from src.mcts.tensor_mcts import TensorMCTS, select_action_gpu
from src.training.replay_buffer import stack_with_history
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from eval_checkpoint_health import build_network


def entropy_bits(p):
    p = np.asarray(p, dtype=float); p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=256, help="parallel start positions")
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda"
    cfg = get_config("chess_small")
    cfg.device = device
    game = ChessGame()
    gpu_game = GpuChessGame()

    print(f"sims={cfg.num_simulations} alpha={cfg.dirichlet_alpha} eps={cfg.dirichlet_epsilon} "
          f"sample_k={cfg.sample_k} temp={args.temp}  N={args.n}")

    for ckpt_path in args.ckpts:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        net = build_network(ckpt, game, cfg, device)

        # Build N identical start-position stacked observations (zero history).
        state = gpu_game.reset_batch(args.n, device=device)
        single = gpu_game.to_tensor_batch(state)  # [N, C, H, W]
        n_frames = cfg.history_frames
        c = single.shape[1]
        obs_window = torch.zeros(args.n, n_frames, c, single.shape[2], single.shape[3], device=device)
        obs_window[:, 0] = single
        stacked = obs_window.reshape(args.n, n_frames * c, single.shape[2], single.shape[3])
        legal_mask = gpu_game.legal_mask(state)  # [N, A]

        mcts = TensorMCTS(net, game, cfg, device=device, compile=False,
                          select_backend="eager")
        torch.manual_seed(0); np.random.seed(0)
        root_data = mcts.run_batch_gpu(stacked, legal_mask, add_noise=True)

        cv = root_data["child_visits"].float()  # [N, K]
        ca = root_data["child_actions"]          # [N, K]
        rv = root_data["root_value"]
        # Per-position visit entropy / top-share (dedup via select_action_gpu target)
        temp_t = torch.tensor(float(args.temp), device=device)
        action, dense_target = select_action_gpu(ca, cv, temp_t, game.action_space_size)

        # dense_target is the per-position deduped visit fraction (temp-independent)
        tgt = dense_target.cpu().numpy()
        per_pos_ent = np.array([entropy_bits(row[row > 0]) for row in tgt])
        per_pos_top = tgt.max(axis=1)

        # Distribution of SAMPLED actions across the N positions (what ply-0 buffer sees)
        acts = action.cpu().numpy()
        # Direct per-position argmax-rate vs top-mass (the buffer-inconsistency test):
        am = (action == torch.from_numpy(np.asarray(tgt).argmax(axis=1)).to(device)).float().mean().item()
        print(f"  [T={args.temp}] per-position argmax_rate={am:.1%}  vs  top-share={per_pos_top.mean():.1%} "
              f"(equal => T=1 sampling consistent)")
        from collections import Counter
        cnt = Counter(int(a) for a in acts)
        tot = sum(cnt.values())
        samp_ent = entropy_bits(np.array(list(cnt.values())))
        top_act, top_n = cnt.most_common(1)[0]
        mv = _action_to_move(int(top_act), chess.Board())

        print(f"\n=== {os.path.basename(ckpt_path)} step={ckpt.get('step')} ===")
        print(f"  per-position visit-target: entropy mean={per_pos_ent.mean():.2f}b  "
              f"top-share mean={per_pos_top.mean():.1%}")
        print(f"  SAMPLED ply-0 across {args.n} pos (temp={args.temp}): "
              f"entropy={samp_ent:.2f}b  top-move={mv.uci() if mv else top_act} "
              f"({top_n/tot:.1%})  unique_moves={len(cnt)}")
        print(f"  mean root_value = {rv.mean().item():+.4f}  std={rv.std().item():.4f}")
        # show top sampled moves
        tops = []
        for a, n in cnt.most_common(5):
            m = _action_to_move(int(a), chess.Board())
            tops.append((m.uci() if m else a, f"{n/tot:.0%}"))
        print(f"  top sampled moves: {tops}")


if __name__ == "__main__":
    main()
