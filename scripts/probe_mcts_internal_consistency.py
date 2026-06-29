"""Probe: MCTS internal consistency on REAL buffer positions.

Loads self-play games from a .buf, samples ~30 positions across game phases
(opening / midgame / pre-repetition), runs the REFERENCE numpy MCTS
(src/mcts/mcts.py, 200 sims) at each, and verifies the search is internally
consistent + sensible. Reports actual per-position numbers.

Invariants checked (each a potential mechanistic bug if violated):
  A. Visit/Q monotonicity: does the MOST-visited child have ~highest Q?
     (rank-correlation of visits vs mover-POV Q across children)
  B. Stored policy target == deduped raw visit distribution? (target alignment)
  C. Value sign/parity STM-correct down the tree: root.value vs child Q signs.
  D. MinMaxStats degenerate (min ~ max)?  span per position.
  E. Root value == visit-weighted child value (mover-POV backup identity)?
  F. Prior vs visits: does search MOVE mass off the prior (search doing work)?
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import GameHistory, stack_with_history
from src.mcts.mcts import MCTS, MinMaxStats, select_action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint_health import build_network


def load_games(buf_path, game, max_games=400):
    import pickle
    games = []
    with open(buf_path, "rb") as f:
        first = pickle.load(f)
        version = first["version"]
        n = first["n_records"]
        for _ in range(n):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if len(games) >= max_games:
                break
    return games


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def mover_q_children(root, discount):
    """Mover-POV Q for each visited child = reward - discount * child_value.
    child_value = child.value_sum / child.visit_count (child POV)."""
    visits = root.child_visits
    vs = root.child_value_sums
    rew = root.child_rewards
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(visits > 0, vs / np.maximum(visits, 1.0), 0.0)
    q = rew - discount * cv
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/chess/2026_06_19_cold2_pc")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit .pt; default = checkpoint matching --buf step")
    ap.add_argument("--buf", default=None,
                    help="explicit .buf; default = checkpoint_30000.buf")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n-positions", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    buf_path = args.buf or os.path.join(args.ckpt_dir, "checkpoint_30000.buf")
    ckpt_path = args.checkpoint or os.path.join(args.ckpt_dir, "checkpoint_30000.pt")

    dev = args.device
    cfg = get_config("chess_small"); cfg.device = dev
    cfg.num_simulations = args.sims
    HF = cfg.history_frames
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    step = ckpt.get("step", "?")
    net = build_network(ckpt, game, cfg, dev)

    print(f"{'='*78}")
    print(f"MCTS INTERNAL-CONSISTENCY PROBE")
    print(f"  ckpt={ckpt_path} step={step}")
    print(f"  buf ={buf_path}")
    print(f"  sims={args.sims}  HF={HF}  discount={cfg.discount}  draw_score={cfg.draw_score}")
    print(f"  sample_k={getattr(cfg,'sample_k',None)}  use_gumbel={getattr(cfg,'use_gumbel',False)}")
    print(f"  dirichlet_alpha={cfg.dirichlet_alpha} eps={cfg.dirichlet_epsilon}")
    print(f"{'='*78}")

    games = load_games(buf_path, game, max_games=600)
    # Self-play games only (no external_values).
    sp = [g for g in games if not g.external_values and len(g.actions) >= 6]
    print(f"loaded {len(games)} games; {len(sp)} self-play (no external_values), "
          f"lengths min/med/max = "
          f"{min(len(g.actions) for g in sp)}/"
          f"{int(np.median([len(g.actions) for g in sp]))}/"
          f"{max(len(g.actions) for g in sp)}")
    rep = sum(1 for g in sp if g.draw_by_repetition)
    nop = sum(1 for g in sp if g.draw_by_no_progress)
    print(f"  draw_by_repetition={rep}  draw_by_no_progress={nop}  (of {len(sp)})")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(sp)

    # Build a pool of (game, ply, phase) samples across phases.
    samples = []
    per_phase = max(1, args.n_positions // 3)
    phase_counts = {"opening": 0, "midgame": 0, "pre-rep": 0}
    for g in sp:
        L = len(g.actions)
        if L < 6:
            continue
        # opening: ply 2-8 ; midgame: middle third ; pre-rep: last 6 plies
        cands = []
        if phase_counts["opening"] < per_phase:
            cands.append(("opening", int(rng.integers(2, min(9, L)))))
        if phase_counts["midgame"] < per_phase:
            mid_lo = L // 3; mid_hi = max(mid_lo + 1, 2 * L // 3)
            cands.append(("midgame", int(rng.integers(mid_lo, mid_hi))))
        if phase_counts["pre-rep"] < per_phase + 2:
            cands.append(("pre-rep", int(rng.integers(max(0, L - 6), L))))
        for phase, ply in cands:
            if ply >= len(g.policies):
                continue
            samples.append((g, ply, phase))
            phase_counts[phase] += 1
        if len(samples) >= args.n_positions:
            break

    print(f"sampled {len(samples)} positions: {phase_counts}\n")

    mcts = MCTS(net, game, cfg, dev)

    # Aggregates
    spearman_vq = []
    target_match_max = []   # max abs diff between stored policy and recomputed visit dist
    rootv_backup_err = []   # |root.value - visit-weighted child mover-Q|
    mm_spans = []
    sibling_q_std = []
    prior_visit_l1 = []     # L1 distance prior vs visit dist (search work)
    rootv_vs_storedrv = []  # |fresh root.value - stored root_value|
    top_is_topq = 0
    parity_ok = 0
    parity_total = 0

    hdr = (f"{'phase':9s} {'ply':>4s} {'nlegal':>6s} {'root.v':>7s} "
           f"{'storRV':>7s} {'mmspan':>7s} {'sibQstd':>8s} {'sp(v,Q)':>8s} "
           f"{'tgtdiff':>8s} {'backupErr':>9s} {'pri|vis':>8s} {'top=topQ':>8s}")
    print(hdr)
    print("-" * len(hdr))

    for (g, ply, phase) in samples:
        # Build the stacked observation with real history at this ply.
        cur = g.observations[ply]
        prior = g.observations[:ply]
        obs = stack_with_history(cur, prior, HF)
        legal = g.legal_actions_list[ply]

        root = mcts.run(obs, legal, add_noise=False)

        visits = root.child_visits
        actions = root.child_actions
        priors = root.child_priors
        q = mover_q_children(root, cfg.discount)
        nlegal = len(actions)

        # --- A: visit/Q rank correlation (only visited children) ---
        visited = visits > 0
        sp_vq = spearman(visits[visited], q[visited]) if visited.sum() >= 2 else float("nan")
        if not np.isnan(sp_vq):
            spearman_vq.append(sp_vq)

        # top-visited child has top Q?
        if visited.sum() >= 1:
            top_v_idx = int(np.argmax(visits))
            top_q_idx = int(np.argmax(np.where(visited, q, -np.inf)))
            top_is_topq += int(top_v_idx == top_q_idx)

        # sibling Q std (the "within-position blindness" metric)
        if visited.sum() >= 2:
            sibling_q_std.append(float(q[visited].std()))

        # --- B: stored policy target vs recomputed visit distribution ---
        _, recomputed = select_action(root, temperature=1.0)
        stored = g.policies[ply]
        # Compare on union of supports up to len of stored.
        m = min(len(recomputed), len(stored))
        diff = np.abs(recomputed[:m].astype(np.float64) - stored[:m].astype(np.float64))
        tgt_diff = float(diff.max()) if len(diff) else float("nan")
        target_match_max.append(tgt_diff)

        # --- C: value parity. root.value is mover POV at root.
        # A visited child's mover-POV Q should be on the same scale; the
        # child's OWN value (child POV) should be ~ -Q/discount-ish sign-flipped.
        # Check: sign of root.value vs visit-weighted child Q (should agree, both mover POV).
        if visited.sum() >= 1:
            vw_q = float(np.average(q[visited], weights=visits[visited]))
            parity_total += 1
            # root.value and vw_q are both mover-POV; in a 1-ply-dominated tree
            # root.value ~= max child Q. Sign agreement is the weak invariant.
            if (root.value >= 0) == (vw_q >= 0) or abs(root.value) < 0.02 or abs(vw_q) < 0.02:
                parity_ok += 1
        else:
            vw_q = float("nan")

        # --- D: MinMaxStats span. Re-run the search bookkeeping to get final mm.
        # MCTS.run builds its own MinMaxStats internally; reconstruct final span
        # from the tree's mover-POV Q range over visited nodes (proxy) — but more
        # directly, recompute from all visited children + root.
        # The actual mm seen = range of (reward - discount*value) over visited nodes.
        node_qs = []
        if visited.sum() >= 1:
            node_qs.extend(list(q[visited]))
        # include root's own backed-up mover-Q proxy (root has reward 0)
        node_qs.append(-cfg.discount * root.value if False else root.value)
        mm_span = (max(node_qs) - min(node_qs)) if len(node_qs) >= 2 else 0.0
        mm_spans.append(mm_span)

        # --- E: root value == visit-weighted child mover-Q (backup identity) ---
        # For MuZero: root.value_sum = V_leaf_root + sum over sims of backed value.
        # The clean identity: root.value ≈ (V_root_init + Σ_children N_c * Q_c) / N_root,
        # but the practical AlphaZero check is root.value ≈ visit-weighted child Q
        # (within ~1/N_root for the root's own leaf eval contribution).
        if not np.isnan(vw_q):
            backup_err = abs(root.value - vw_q)
            rootv_backup_err.append(backup_err)
        else:
            backup_err = float("nan")

        # --- F: prior vs visit distribution (is search doing work?) ---
        vsum = visits.sum()
        vis_dist = visits / vsum if vsum > 0 else np.zeros_like(visits)
        pv_l1 = float(np.abs(vis_dist - priors).sum())
        prior_visit_l1.append(pv_l1)

        # stored root value
        stored_rv = float(g.root_values[ply]) if ply < len(g.root_values) else float("nan")
        rootv_vs_storedrv.append(abs(root.value - stored_rv))

        top_flag = "Y" if (visited.sum() >= 1 and top_v_idx == top_q_idx) else "n"
        print(f"{phase:9s} {ply:4d} {nlegal:6d} {root.value:+7.3f} "
              f"{stored_rv:+7.3f} {mm_span:7.3f} "
              f"{(sibling_q_std[-1] if (visited.sum()>=2) else float('nan')):8.4f} "
              f"{sp_vq:8.3f} {tgt_diff:8.4f} {backup_err:9.4f} {pv_l1:8.3f} {top_flag:>8s}")

    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    def stats(name, arr, fmt="{:.4f}"):
        arr = [a for a in arr if not (isinstance(a, float) and np.isnan(a))]
        if not arr:
            print(f"  {name:34s} (no data)")
            return
        a = np.array(arr, dtype=np.float64)
        print(f"  {name:34s} mean={fmt.format(a.mean())} "
              f"med={fmt.format(np.median(a))} "
              f"min={fmt.format(a.min())} max={fmt.format(a.max())} (n={len(a)})")
    stats("A spearman(visits,Q)", spearman_vq)
    print(f"  A top-visited==top-Q child:         {top_is_topq}/{len(samples)}")
    stats("A sibling mover-Q std", sibling_q_std)
    stats("B target diff (stored vs recomputed)", target_match_max)
    print(f"  C value-sign parity ok:             {parity_ok}/{parity_total}")
    stats("D MinMaxStats span (Q range)", mm_spans)
    stats("E root.v vs visit-wtd child-Q err", rootv_backup_err)
    stats("F L1(prior, visit-dist)", prior_visit_l1)
    stats("  fresh root.v vs STORED root_value", rootv_vs_storedrv)

    print("\nINTERPRETATION GUIDE:")
    print("  A near +1 = visits track Q (healthy). near 0/neg = visits ignore Q (BUG/degenerate).")
    print("  B should be ~0 (stored target == temperature-indep visit dist). Large = MISALIGN BUG.")
    print("  C should be ~all-ok. Failures = sign/parity bug down the tree.")
    print("  D near 0 = MinMaxStats degenerate (no Q resolution) -> PUCT loses value term.")
    print("  E should be small (backup identity). Large = backup/POV bug.")
    print("  F large = search moves mass off prior (doing work). ~0 = visits==prior (no search effect).")


if __name__ == "__main__":
    main()
