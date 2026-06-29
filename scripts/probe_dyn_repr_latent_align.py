"""Does the DYNAMICS latent align with the REPRESENTATION latent of the real
next state? (Is the consistency loss actually doing its job?)

For real root positions, for each legal action a:
  h_dyn(a)  = dynamics(repr(root_obs), a)      [latent MCTS uses at depth 1]
  h_repr(a) = repr(real_board_after_a)         [the "true" latent]
Report cosine(h_dyn(a), h_repr(a)) per action, AND the cross-action structure:
  - does h_dyn even VARY across actions? (std of flattened latent across a)
  - cosine between h_dyn(a) and h_repr(a) for the SAME a vs DIFFERENT a'
If same-a cosine ~ different-a' cosine, dynamics is action-blind in latent space.

Also: value(h_dyn) vs value(h_repr) per action (same as dyn_value_blind but
paired with the latent alignment for diagnosis).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer, stack_with_history
from scripts.eval_checkpoint_health import build_network


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.pt")
    ap.add_argument("--buf", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-actions", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    dev = args.device
    buf_path = args.buf or (os.path.splitext(args.checkpoint)[0] + ".buf")
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    HF = cfg.history_frames
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    print(f"loaded step={ckpt.get('step')}", flush=True)
    buf = ReplayBuffer(max_size=10000); buf.load(buf_path, game=game)
    print(f"buffer {len(buf.buffer)} games", flush=True)
    rng = np.random.default_rng(args.seed)
    games = [g for g in buf.buffer if len(g.actions) >= 12]

    same_a_cos, diff_a_cos, dyn_latent_action_std = [], [], []
    for trial in range(args.n):
        g = games[int(rng.integers(len(games)))]
        ply = int(rng.integers(0, len(g.actions) - 1))
        obs = g._stack_history(ply, HF)
        legal = g.legal_actions_list[ply]
        st = game.reset()
        for a in g.actions[:ply]:
            st, _, _ = game.step(st, a)
        board = st.board
        prior_obs = g.observations[: ply + 1]

        hidden = net.representation(obs.unsqueeze(0).to(dev))
        probe = [int(a) for a in legal[:args.max_actions]
                 if _action_to_move(int(a), board) is not None
                 and _action_to_move(int(a), board) in board.legal_moves]
        if len(probe) < 3:
            continue
        acts = torch.tensor(probe, device=dev)
        hn = hidden.expand(len(probe), *hidden.shape[1:]).contiguous()
        h_dyn, _ = net.dynamics(hn, acts)          # (A, C, H, W)
        h_dyn_flat = h_dyn.reshape(len(probe), -1)

        # repr of real next boards
        h_repr_list = []
        for a in probe:
            mv = _action_to_move(a, board)
            nb = board.copy(); nb.push(mv)
            nst = game.reset(); nst.board = nb
            nobs = stack_with_history(game.to_tensor(nst), prior_obs, HF)
            h_repr_list.append(net.representation(nobs.unsqueeze(0).to(dev)).reshape(-1))
        h_repr_flat = torch.stack(h_repr_list)     # (A, D)

        dn = F.normalize(h_dyn_flat, dim=-1)
        rn = F.normalize(h_repr_flat, dim=-1)
        cos_mat = dn @ rn.T                          # (A, A): [i,j]=cos(h_dyn_i, h_repr_j)
        A = len(probe)
        same = float(torch.diag(cos_mat).mean())
        off = cos_mat[~torch.eye(A, dtype=bool)].mean().item()
        # how much does h_dyn vary across actions?
        ldas = float(h_dyn_flat.std(0).mean())
        same_a_cos.append(same); diff_a_cos.append(off); dyn_latent_action_std.append(ldas)
        print(f"[{trial:2d}] ply={ply:3d} A={A:2d} | cos(h_dyn_a, h_repr_a) same={same:+.3f} "
              f"diff-action={off:+.3f} gap={same-off:+.3f} | h_dyn per-dim std across actions={ldas:.4f}",
              flush=True)

    print("\n" + "=" * 90)
    sa = np.array(same_a_cos); da = np.array(diff_a_cos)
    print(f"SUMMARY over {len(sa)} positions:")
    print(f"  cos(h_dyn(a), h_repr(a))  SAME action: mean={sa.mean():+.3f}")
    print(f"  cos(h_dyn(a), h_repr(a')) DIFF action: mean={da.mean():+.3f}")
    print(f"  alignment GAP (same - diff): {sa.mean()-da.mean():+.3f}  "
          f"(>0 => dynamics latent is action-specific & aligned to real next; "
          f"~0 => action-blind / unaligned in latent space)")
    print(f"  h_dyn per-dim std across actions: mean={np.mean(dyn_latent_action_std):.4f}")


if __name__ == "__main__":
    main()
