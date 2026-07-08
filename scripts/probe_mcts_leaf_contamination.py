"""Probe: leaf-node FULL-action-space expansion — does it contaminate the search?

Reference MCTS expands non-root leaves with the FULL 4672-action space
(mcts.py:175-176), masking only at the root. Two questions:
  1. How DEEP does the tree actually go at 200 sims? (if depth ~= 1, the
     leaf contamination is mostly irrelevant — root children rarely re-expanded.)
  2. When a depth>=1 child IS selected for re-expansion, what fraction of its
     grandchild visits land on ILLEGAL moves (latent states the dynamics net
     was never trained on)?
  3. How much policy-net prior mass sits on ILLEGAL actions at the root latent
     (initial_inference) vs at a 1-ply child latent (recurrent_inference)?
     -> if illegal mass BALLOONS at depth 1, the dynamics+policy is OOD there.

Instrumented by monkeypatching MCTS._expand to record (depth, legal_set) and
walking the final tree.
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
from src.training.replay_buffer import GameHistory, stack_with_history
from src.mcts.mcts import MCTS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint_health import build_network


def load_games(buf_path, game, max_games=200):
    import pickle
    games = []
    with open(buf_path, "rb") as f:
        first = pickle.load(f)
        version = first["version"]; n = first["n_records"]
        for _ in range(n):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if len(games) >= max_games:
                break
    return games


def tree_stats(root):
    """Walk the materialized tree. Returns max depth, depth histogram of visits,
    and total visits."""
    depth_visits = {}
    max_depth = 0
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        max_depth = max(max_depth, d)
        depth_visits[d] = depth_visits.get(d, 0) + node.visit_count
        for c in node.children:
            if c is not None and c.visit_count > 0:
                stack.append((c, d + 1))
    return max_depth, depth_visits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/chess/2026_06_19_cold2_pc")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n-positions", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    buf_path = os.path.join(args.ckpt_dir, "checkpoint_30000.buf")
    ckpt_path = os.path.join(args.ckpt_dir, "checkpoint_30000.pt")
    dev = args.device
    cfg = get_config("chess_small"); cfg.device = dev; cfg.num_simulations = args.sims
    HF = cfg.history_frames
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    mcts = MCTS(net, game, cfg, dev)

    games = load_games(buf_path, game, max_games=200)
    sp = [g for g in games if not g.external_values and len(g.actions) >= 10]
    rng = np.random.default_rng(args.seed); rng.shuffle(sp)

    samples = []
    for g in sp:
        L = len(g.actions); ply = int(rng.integers(2, L - 1))
        if ply < len(g.policies):
            samples.append((g, ply))
        if len(samples) >= args.n_positions:
            break

    print(f"sims={args.sims} positions={len(samples)} action_space={game.action_space_size}", flush=True)
    print(f"{'ply':>4s} {'nleg':>4s} {'maxD':>4s} {'visD0':>6s} {'visD1':>6s} "
          f"{'visD2+':>6s} {'illegPriRoot':>12s} {'illegPriChild':>13s} {'illegVisD1':>10s}", flush=True)
    print("-" * 90, flush=True)

    agg = dict(maxd=[], illroot=[], illchild=[], illvisd1=[], d2frac=[])
    for (g, ply) in samples:
        cur = g.observations[ply]; prior = g.observations[:ply]
        obs = stack_with_history(cur, prior, HF)
        legal = g.legal_actions_list[ply]
        legal_set = set(int(a) for a in legal)

        root = mcts.run(obs, legal, add_noise=False)

        maxd, dv = tree_stats(root)
        tot = sum(dv.values())
        vd0 = dv.get(0, 0); vd1 = dv.get(1, 0)
        vd2 = sum(v for d, v in dv.items() if d >= 2)

        # Illegal prior mass at root latent: softmax over ALL actions of
        # initial_inference policy logits, summed over illegal.
        with torch.no_grad():
            h, pl_root, _ = net.initial_inference(obs.unsqueeze(0).to(dev))
            p_root = F.softmax(pl_root.squeeze(0).float(), dim=0).cpu().numpy()
        ill_root = float(sum(p_root[a] for a in range(len(p_root)) if a not in legal_set))

        # Illegal prior at a depth-1 child latent: take the most-visited root child,
        # run recurrent_inference, softmax over all actions. Legal set at depth 1 =
        # legal moves of the actual resulting board (compute via python-chess).
        ill_child = float("nan"); ill_vis_d1 = float("nan")
        if root.child_visits.max() > 0:
            top_idx = int(np.argmax(root.child_visits))
            top_action = int(root.child_actions[top_idx])
            # Resulting board legal set
            board = g.observations  # placeholder; recompute board by replay
            # Rebuild the board at this ply, push the move, get legal set.
            from src.games.chess import _action_to_move
            st = game.reset()
            # replay to ply
            for a in g.actions[:ply]:
                st, _, _ = game.step(st, a)
            mv = _action_to_move(top_action, st.board)
            child_legal_set = None
            if mv is not None and mv in st.board.legal_moves:
                nb = st.board.copy(); nb.push(mv)
                child_legal_set = set(
                    int(__import__("src.games.chess", fromlist=["_move_to_action"])._move_to_action(m, nb.turn))
                    for m in nb.legal_moves
                )
            with torch.no_grad():
                a_t = torch.tensor([top_action], device=dev)
                nh, rew, pl_c, val_c = net.recurrent_inference(h, a_t)
                p_child = F.softmax(pl_c.squeeze(0).float(), dim=0).cpu().numpy()
            if child_legal_set is not None:
                ill_child = float(sum(p_child[a] for a in range(len(p_child)) if a not in child_legal_set))
                # Fraction of the top child's grandchild visits on illegal moves:
                tc = root.children[top_idx]
                if tc is not None and tc.child_visits is not None and tc.child_visits.sum() > 0:
                    gv = tc.child_visits; ga = tc.child_actions
                    ill_v = sum(gv[i] for i in range(len(ga)) if int(ga[i]) not in child_legal_set)
                    ill_vis_d1 = float(ill_v / gv.sum())

        d2frac = vd2 / tot if tot > 0 else 0.0
        print(f"{ply:4d} {len(legal):4d} {maxd:4d} {vd0:6d} {vd1:6d} {vd2:6d} "
              f"{ill_root:12.4f} {ill_child:13.4f} {ill_vis_d1:10.4f}", flush=True)
        agg["maxd"].append(maxd); agg["illroot"].append(ill_root)
        agg["illchild"].append(ill_child); agg["illvisd1"].append(ill_vis_d1)
        agg["d2frac"].append(d2frac)

    print("\nAGGREGATE")
    def s(name, key, fmt="{:.4f}"):
        a = np.array([x for x in agg[key] if not (isinstance(x, float) and np.isnan(x))], dtype=np.float64)
        if len(a) == 0:
            print(f"  {name:34s} (no data)"); return
        print(f"  {name:34s} mean={fmt.format(a.mean())} med={fmt.format(np.median(a))} "
              f"min={fmt.format(a.min())} max={fmt.format(a.max())}")
    s("max tree depth", "maxd", "{:.1f}")
    s("frac visits at depth>=2", "d2frac", "{:.4f}")
    s("illegal PRIOR mass @ root latent", "illroot")
    s("illegal PRIOR mass @ depth-1 latent", "illchild")
    s("illegal VISIT frac @ depth-1 child", "illvisd1")
    print("\n  If depth ~1 and d2frac~0: leaf full-expansion barely matters (tree too shallow).")
    print("  If illegal prior BALLOONS at depth 1: dynamics->policy is OOD off the root.")


if __name__ == "__main__":
    main()
