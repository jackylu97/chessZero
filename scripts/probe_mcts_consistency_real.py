"""Internal-consistency audit of numpy MCTS (src/mcts/mcts.py) on REAL buffer
positions across game phases.

For ~N real positions sampled from a checkpoint .buf (opening / midgame /
pre-repetition), run MCTS.run (200 sims, reference numpy engine) and verify the
search is internally consistent + sensible. Checks per position:

  C1  visits track Q: rank-correlation between child visit counts and child mover-POV Q
  C2  stored policy target == deduped, temperature-independent visit distribution
      (compares select_action(root,T) raw probs across T in {0.0,1.0,2.0})
  C3  value sign / parity STM-correct down the tree (spot-check a path)
  C4  MinMaxStats degeneracy: min vs max span after search
  C5  root value == visit-weighted child value identity:
        root.value ?= sum_a N(a)/N * (reward_a - gamma*child_a.value)   (mover POV)
      (the backup identity for a tree: root.value_sum/visits)
  C6  flat-Q symptom: std of top-children Q

Also dumps the actual per-position numbers.

CPU by default. Reuses build_network from eval_checkpoint_health.
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
from src.mcts.mcts import MCTS, select_action
from scripts.eval_checkpoint_health import build_network


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def child_q_mover(root, idx, discount):
    """Mover-POV Q at child idx (matches _select_child)."""
    v = root.child_visits[idx]
    if v <= 0:
        return None
    cv = root.child_value_sums[idx] / v  # child's own POV
    return float(root.child_rewards[idx] - discount * cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.pt")
    ap.add_argument("--buf", default=None, help="defaults to checkpoint's .buf")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n-positions", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    buf_path = args.buf or (os.path.splitext(args.checkpoint)[0] + ".buf")
    torch.serialization.add_safe_globals([MuZeroConfig])

    game = ChessGame()
    cfg = get_config("chess_small")
    cfg.device = dev
    HF = cfg.history_frames
    print(f"config: sims(used)={args.sims} sample_k={cfg.sample_k} use_gumbel={cfg.use_gumbel} "
          f"draw_score={cfg.draw_score} discount={cfg.discount} HF={HF} "
          f"dirichlet_alpha={cfg.dirichlet_alpha} moves_left_mcts={cfg.moves_left_mcts}")

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    print(f"loaded {args.checkpoint} step={ckpt.get('step')}")

    buf = ReplayBuffer(max_size=10000)
    buf.load(buf_path, game=game)
    print(f"loaded {buf_path}: {len(buf.buffer)} games")

    rng = np.random.default_rng(args.seed)

    # Sample positions across phases. Pick games, then sample plies in
    # opening (<=12), midgame (12..len-12), pre-repetition (last ~10 for
    # threefold games — where the shuffle happens).
    rep_games = [g for g in buf.buffer if g.draw_by_repetition and len(g.actions) >= 20]
    other_games = [g for g in buf.buffer if not g.draw_by_repetition and len(g.actions) >= 12]
    print(f"rep games: {len(rep_games)}  non-rep games(>=12 ply): {len(other_games)}")

    positions = []  # (label, game_history, ply)

    def add_from(games, n, phase):
        if not games:
            return
        gi = rng.choice(len(games), size=min(n, len(games)), replace=len(games) < n)
        for g_idx in gi:
            g = games[int(g_idx)]
            L = len(g.actions)
            if phase == "opening":
                ply = int(rng.integers(0, min(12, L)))
            elif phase == "midgame":
                lo, hi = min(12, L - 1), max(13, L - 12)
                ply = int(rng.integers(lo, max(lo + 1, hi)))
            elif phase == "prerep":
                ply = int(rng.integers(max(0, L - 10), L))
            ply = min(ply, L - 1)
            positions.append((phase, g, ply))

    n_each = args.n_positions // 3
    add_from(other_games, n_each, "opening")
    add_from(other_games, n_each, "midgame")
    add_from(rep_games, args.n_positions - 2 * n_each, "prerep")

    cfg.num_simulations = args.sims
    mcts = MCTS(net, game, cfg, dev)

    # Aggregate stats
    agg = {"visit_q_spearman": [], "q_spread_top": [], "mm_span": [],
           "root_minus_vwq": [], "policy_target_match": [], "policy_T_indep": [],
           "root_value": [], "n_legal": [], "n_visited": []}

    print("\n" + "=" * 110)
    for pi, (phase, g, ply) in enumerate(positions):
        obs = g._stack_history(ply, HF)
        legal = g.legal_actions_list[ply] if ply < len(g.legal_actions_list) else game.legal_actions(game.reset())
        # stored policy target at this ply (dense over action space)
        stored_policy = g.policies[ply] if ply < len(g.policies) else None
        stored_rv = g.root_values[ply] if ply < len(g.root_values) else None
        reana = getattr(g, "reanalyze_count", 0)

        root = mcts.run(obs, legal, add_noise=False)

        # ---- gather child stats ----
        visits = root.child_visits.copy()
        actions = root.child_actions.copy()
        qs = np.array([child_q_mover(root, i, cfg.discount) if visits[i] > 0 else np.nan
                       for i in range(len(actions))], dtype=np.float64)
        visited = visits > 0
        nvis = int(visited.sum())

        # C1: visits track Q (only over visited children)
        if nvis >= 2:
            sp = spearman(visits[visited], qs[visited])
        else:
            sp = float("nan")
        agg["visit_q_spearman"].append(sp)

        # C6: Q spread among top-visited children
        order = np.argsort(-visits)
        topk = [i for i in order if visits[i] > 0][:6]
        top_qs = qs[topk]
        q_spread = float(np.nanstd(top_qs)) if len(top_qs) >= 2 else float("nan")
        agg["q_spread_top"].append(q_spread)

        # C4: MinMaxStats span — reconstruct from final child Qs + root
        # (the search's own mm is internal; we approximate the realized Q range)
        all_q = qs[visited]
        mm_span = float(np.nanmax(all_q) - np.nanmin(all_q)) if len(all_q) >= 2 else 0.0
        agg["mm_span"].append(mm_span)

        # C5: root.value vs visit-weighted child mover-POV Q
        # Tree backup identity: root.value_sum/visits should equal
        # (sum over children of N(a)*(reward - gamma*child.value) ) / (visits-? )
        # Exact MuZero identity: root.value*N_root = V_leaf_root + sum_children N(a)*Q_mover(a)
        # We test the practical identity used downstream: visit-weighted child Q ≈ root.value.
        if nvis >= 1:
            w = visits[visited] / visits[visited].sum()
            vwq = float(np.sum(w * qs[visited]))
        else:
            vwq = float("nan")
        agg["root_minus_vwq"].append(root.value - vwq)

        # C2: stored policy target == deduped temperature-independent visit distribution
        _, probs_T1 = select_action(root, temperature=1.0)
        _, probs_T0 = select_action(root, temperature=0.0)
        _, probs_T2 = select_action(root, temperature=2.0)
        # temperature-independence of the RETURNED TARGET (should be identical raw fracs)
        t_indep = float(np.max(np.abs(probs_T1 - probs_T0))) + float(np.max(np.abs(probs_T1 - probs_T2)))
        agg["policy_T_indep"].append(t_indep)

        # compare to stored target (only meaningful if NOT reanalyzed and net unchanged;
        # here net == current checkpoint so for reanalyzed-with-this-ckpt it'd match.
        # We report the L1 distance regardless — large == stored target came from a
        # different net/engine (expected: self-play used BatchedMCTS w/ sample_k+noise).
        if stored_policy is not None and stored_policy.sum() > 0:
            ml = max(len(stored_policy), len(probs_T1))
            a1 = np.zeros(ml); a2 = np.zeros(ml)
            a1[:len(stored_policy)] = stored_policy
            a2[:len(probs_T1)] = probs_T1
            pol_l1 = float(np.abs(a1 - a2).sum())
        else:
            pol_l1 = float("nan")
        agg["policy_target_match"].append(pol_l1)

        agg["root_value"].append(root.value)
        agg["n_legal"].append(len(legal))
        agg["n_visited"].append(nvis)

        # ---- per-position dump (first 12, then sparse) ----
        if pi < 12 or pi % 5 == 0:
            board = None
            try:
                # rebuild board to get SAN
                st = game.reset()
                for a in g.actions[:ply]:
                    st, _, _ = game.step(st, a)
                board = st.board
            except Exception:
                board = None
            top_str = []
            for i in order[:4]:
                a = int(actions[i]); vis = visits[i]; q = qs[i]
                san = f"a{a}"
                if board is not None:
                    mv = _action_to_move(a, board)
                    if mv is not None and mv in board.legal_moves:
                        san = board.san(mv)
                pr = float(root.child_priors[i])
                top_str.append(f"{san}(N={vis:.0f},Q={q:+.3f},p={pr:.3f})")
            print(f"[{pi:2d}] {phase:8s} ply={ply:3d}/{len(g.actions):3d} reana={reana} "
                  f"legal={len(legal):3d} visited={nvis:3d} rootV={root.value:+.4f} "
                  f"storedRV={stored_rv if stored_rv is None else round(stored_rv,4)}")
            print(f"      top: {'  '.join(top_str)}")
            print(f"      C1 visit~Q spearman={sp:+.3f} | C6 topQ_spread={q_spread:.4f} | "
                  f"C4 mm_span={mm_span:.4f} | C5 rootV-vwQ={root.value - vwq:+.5f} | "
                  f"C2 T_indep={t_indep:.2e} stored_vs_freshL1={pol_l1 if np.isnan(pol_l1) else round(pol_l1,3)}")

    # ---- summary ----
    def stat(name, vals, fmt="{:+.4f}"):
        v = np.array(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            print(f"  {name:28s}  (no finite values)")
            return
        print(f"  {name:28s}  mean={fmt.format(v.mean())}  median={fmt.format(np.median(v))}  "
              f"min={fmt.format(v.min())}  max={fmt.format(v.max())}  n={len(v)}")

    print("\n" + "=" * 110)
    print(f"SUMMARY over {len(positions)} positions ({args.sims} sims, numpy MCTS, no noise)")
    print("=" * 110)
    stat("C1 visit~Q spearman", agg["visit_q_spearman"])
    stat("C6 topQ spread (flat=draw)", agg["q_spread_top"])
    stat("C4 MinMax realized span", agg["mm_span"])
    stat("C5 rootV - visitWeightedQ", agg["root_minus_vwq"], "{:+.6f}")
    stat("C2 policy T-independence", agg["policy_T_indep"], "{:.2e}")
    stat("C2 stored-vs-fresh policy L1", agg["policy_target_match"], "{:.3f}")
    stat("root value", agg["root_value"])
    stat("n_legal", agg["n_legal"], "{:.0f}")
    stat("n_visited (of legal)", agg["n_visited"], "{:.0f}")

    # interpretive flags
    print("\nINTERPRETATION:")
    sp_m = np.nanmean(agg["visit_q_spearman"])
    qsp_m = np.nanmean(agg["q_spread_top"])
    c5_m = np.nanmean(np.abs(agg["root_minus_vwq"]))
    tindep_m = np.nanmax(agg["policy_T_indep"])
    print(f"  visit~Q corr {sp_m:+.3f}  (expect >0 if visits follow Q; near 0/neg = pathological)")
    print(f"  top-Q spread {qsp_m:.4f}  (flat-Q draw-basin symptom if << 0.05)")
    print(f"  |rootV - visitWeightedQ| {c5_m:.5f}  (large = backup identity broken)")
    print(f"  max policy-target T-dependence {tindep_m:.2e}  (>1e-6 = TARGET depends on temperature -> BUG)")


if __name__ == "__main__":
    main()
