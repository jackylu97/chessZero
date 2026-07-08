"""Is the DYNAMICS value head action-blind? Compare, per real root position:

  - dynNet_V(a): value head on the 1-step dynamics latent for each legal action a
  - GT_V(a):     value head on representation(real board after playing a)   [ground truth]
  - reprNext separation: std over actions of GT_V(a)  (true value spread across moves)
  - dynNet separation:   std over actions of dynNet_V(a) (modeled spread)

If dynNet std << GT std, the dynamics world-model collapses sibling values to a
constant -> MCTS sibling-Q has no signal even though the TRUE positions differ.
This would be a MECHANISTIC cause of "no within-position signal" distinct from
the value head itself being globally flat (the across-position V^pi story).

Also reports corr(dynNet_V(a), GT_V(a)) across actions per position: does the
dynamics value even RANK siblings the way the real value head would?
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

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
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-actions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
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

    dyn_stds, gt_stds, corrs, dyn_means, gt_means = [], [], [], [], []
    dyn_gt_l1 = []
    for trial in range(args.n):
        g = games[int(rng.integers(len(games)))]
        ply = int(rng.integers(0, len(g.actions) - 1))
        obs = g._stack_history(ply, HF)
        legal = g.legal_actions_list[ply]
        st = game.reset()
        for a in g.actions[:ply]:
            st, _, _ = game.step(st, a)
        board = st.board

        hidden, _, root_v = net.initial_inference(obs.unsqueeze(0).to(dev))
        root_v = float(root_v.item())
        # batch dynamics over up to max_actions legal actions
        probe = legal[:args.max_actions]
        acts = torch.tensor(probe, device=dev)
        hn = hidden.expand(len(probe), *hidden.shape[1:]).contiguous()
        _, _, _, dyn_v = net.recurrent_inference(hn, acts)
        dyn_v = dyn_v.view(-1).cpu().numpy()

        # ground truth: repr(real next board) value, per action — with PROPER
        # 8-frame history (prior frames = the game's observations up to ply).
        # prior_obs is chronological, newest at the end, == observations[:ply+1].
        prior_obs = g.observations[: ply + 1]  # frames at plies 0..ply (current at end)
        gt_v = []
        keep = []
        for k, a in enumerate(probe):
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                continue
            nb = board.copy(); nb.push(mv)
            nst = game.reset(); nst.board = nb
            nobs = stack_with_history(game.to_tensor(nst), prior_obs, HF)
            _, _, v = net.initial_inference(nobs.unsqueeze(0).to(dev))
            gt_v.append(float(v.item())); keep.append(k)
        gt_v = np.array(gt_v)
        dyn_v_k = dyn_v[keep]
        if len(gt_v) < 3:
            continue
        ds = float(dyn_v_k.std()); gs = float(gt_v.std())
        c = float(np.corrcoef(dyn_v_k, gt_v)[0, 1]) if ds > 1e-9 and gs > 1e-9 else float("nan")
        dyn_stds.append(ds); gt_stds.append(gs); corrs.append(c)
        dyn_means.append(float(dyn_v_k.mean())); gt_means.append(float(gt_v.mean()))
        dyn_gt_l1.append(float(np.mean(np.abs(dyn_v_k - gt_v))))
        print(f"[{trial:2d}] ply={ply:3d} nacts={len(gt_v):2d} rootV={root_v:+.3f} | "
              f"dynNet: mean={dyn_v_k.mean():+.3f} std={ds:.4f} | "
              f"GT_repr: mean={gt_v.mean():+.3f} std={gs:.4f} | "
              f"corr(dyn,GT)={c:+.3f}", flush=True)

    print("\n" + "=" * 90)
    da = np.array(dyn_stds); ga = np.array(gt_stds); ca = np.array(corrs)
    ca = ca[np.isfinite(ca)]
    print(f"SUMMARY over {len(dyn_stds)} positions:")
    print(f"  dynNet value std across siblings: mean={da.mean():.4f} median={np.median(da):.4f}")
    print(f"  GT(repr) value std across siblings: mean={ga.mean():.4f} median={np.median(ga):.4f}")
    print(f"  ratio dynNet_std / GT_std: {da.mean()/max(ga.mean(),1e-9):.3f}  "
          f"(<<1 = dynamics collapses sibling value spread)")
    print(f"  corr(dynNet_V, GT_V) across siblings: mean={ca.mean():+.3f} median={np.median(ca):+.3f} "
          f"(near 0 = dynamics value does NOT rank siblings like the real value head)")
    print(f"  dynNet mean: {np.mean(dyn_means):+.4f}  GT mean: {np.mean(gt_means):+.4f}")
    print(f"  mean |dynNet_V - GT_V|: {np.mean(dyn_gt_l1):.4f}")


if __name__ == "__main__":
    main()
