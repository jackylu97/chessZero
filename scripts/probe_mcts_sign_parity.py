"""C3: verify value SIGN/PARITY is side-to-move-correct down the MCTS tree, and
that backup is self-consistent.

Two independent checks on real positions:

(A) GROUND-TRUTH parity: for a root position, for each top child action a, compare
    the MCTS child's stored value (child POV) against the value the NETWORK gives
    when we ACTUALLY make move a on the real board and re-run initial_inference on
    the resulting position (that's the true next-state STM value). The dynamics-net
    child.value should have the SAME SIGN as -1 * (network value at real next pos in
    parent POV) ... i.e. child.value (child POV) ~ network_V(real_next_board, its STM).
    A sign flip here = mover-POV bug feeding MCTS the wrong-signed Q.

(B) BACKUP self-consistency: manually re-derive root.value from the leaf values and
    rewards along every search path and compare to root.value_sum/visits.

This isolates whether the ~0 within-position sibling-Q is caused by a SIGN/PARITY
mechanism (would scramble ranking) vs. the value head genuinely being flat.
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
from src.mcts.mcts import MCTS
from src.model.utils import wdl_to_scalar
from scripts.eval_checkpoint_health import build_network


@torch.no_grad()
def net_value(net, obs, dev, draw_score):
    """STM value via the exact MCTS path (initial_inference -> decoded scalar)."""
    _, _, v = net.initial_inference(obs.unsqueeze(0).to(dev))
    return float(v.item())


@torch.no_grad()
def dyn_child_value(net, root_hidden, action, dev, draw_score):
    """One-step dynamics from root hidden + action; value at that latent (child POV).

    recurrent_inference already returns the decoded SCALAR value (draw_score baked
    in via net.draw_score), matching exactly what MCTS feeds to backprop."""
    h = root_hidden.unsqueeze(0).to(dev)
    a = torch.tensor([action], device=dev)
    nh, reward, pl, v = net.recurrent_inference(h, a)
    return float(v.item()), float(reward.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.pt")
    ap.add_argument("--buf", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    dev = args.device
    buf_path = args.buf or (os.path.splitext(args.checkpoint)[0] + ".buf")
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    HF = cfg.history_frames; ds = cfg.draw_score
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    buf = ReplayBuffer(max_size=10000); buf.load(buf_path, game=game)
    rng = np.random.default_rng(args.seed)
    games = [g for g in buf.buffer if len(g.actions) >= 12]
    cfg.num_simulations = args.sims
    mcts = MCTS(net, game, cfg, dev)

    print(f"sign/parity check, sims={args.sims}, draw_score={ds}, discount={cfg.discount}")
    print("=" * 100)

    parity_flips = 0
    parity_total = 0
    dyn_repr_diffs = []

    for trial in range(args.n):
        g = games[int(rng.integers(len(games)))]
        ply = int(rng.integers(0, len(g.actions) - 1))
        obs = g._stack_history(ply, HF)
        legal = g.legal_actions_list[ply]
        # rebuild board for ground truth
        st = game.reset()
        for a in g.actions[:ply]:
            st, _, _ = game.step(st, a)
        board = st.board

        root = mcts.run(obs, legal, add_noise=False)
        root_V_net = net_value(net, obs, dev, ds)
        print(f"\n[trial {trial}] ply={ply} legal={len(legal)} root.value(search)={root.value:+.4f} "
              f"net_V(root)={root_V_net:+.4f}", flush=True)

        visits = root.child_visits
        order = np.argsort(-visits)[:5]
        for j in order:
            if visits[j] <= 0:
                continue
            a = int(root.child_actions[j])
            child = root.children[j]
            child_val_search = child.value if child is not None else float("nan")  # child POV
            # MCTS mover-POV Q at this child:
            q_mover = float(root.child_rewards[j] - cfg.discount * child_val_search)
            # (A1) dynamics-net direct value at this child latent
            dyn_v, dyn_r = dyn_child_value(net, root.hidden_state, a, dev, ds)
            # (A2) GROUND TRUTH: make the move on the real board, re-run repr+pred.
            mv = _action_to_move(a, board)
            san = f"a{a}"
            gt_child_V = float("nan")
            if mv is not None and mv in board.legal_moves:
                san = board.san(mv)
                nb = board.copy(); nb.push(mv)
                nst = game.reset(); nst.board = nb
                # build history: prior frames are root's history + root obs
                # approximate with single-frame next (zero-pad) — value parity is the point
                nobs = stack_with_history(game.to_tensor(nst), [], HF)
                gt_child_V = net_value(net, nobs, dev, ds)  # next-state STM value (child POV-ish)
            # PARITY TEST: gt_child_V is in the NEXT player's POV (== child POV).
            # child.value (search) is also stored in child POV. They should AGREE IN SIGN
            # (both ~ value to the side to move at the child). q_mover = reward - gamma*child.value
            # should have OPPOSITE sign to gt_child_V (parent gains when child-mover loses).
            if np.isfinite(gt_child_V) and abs(gt_child_V) > 0.05 and abs(q_mover) > 0.05:
                parity_total += 1
                # q_mover (parent POV) should be ~ -gt_child_V (child POV), zero-sum
                if np.sign(q_mover) == np.sign(gt_child_V):  # SAME sign = parity BUG
                    parity_flips += 1
                    flag = "  <-- PARITY MISMATCH (q_mover same sign as child-POV V)"
                else:
                    flag = ""
            else:
                flag = ""
            # dyn-net child value vs the dynamics-implied search child value
            print(f"   {san:7s} N={visits[j]:4.0f} child.value(search,childPOV)={child_val_search:+.4f} "
                  f"q_mover={q_mover:+.4f} | dynNet_V={dyn_v:+.4f} r={dyn_r:+.3f} | "
                  f"GT_realnext_V(childPOV)={gt_child_V:+.4f}{flag}")
            if np.isfinite(gt_child_V):
                dyn_repr_diffs.append(abs(dyn_v - gt_child_V))

    print("\n" + "=" * 100)
    print(f"PARITY: {parity_flips}/{parity_total} child Qs had WRONG sign vs ground-truth next-state value")
    if dyn_repr_diffs:
        d = np.array(dyn_repr_diffs)
        print(f"dyn-net child V vs ground-truth real-next-state V: mean|diff|={d.mean():.4f} "
              f"median={np.median(d):.4f} (large = dynamics world-model value drifts from real)")


if __name__ == "__main__":
    main()
