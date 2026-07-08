"""Follow-up probe: WHY do visits not track Q?  PUCT term balance + noise/engine
confound check for the stored-policy-target mismatch.

For ~20 real buffer positions, run reference numpy MCTS (200 sims) and:
  1. At the FINAL root, decompose the PUCT score of each visited child into
     prior_score vs value_score. If prior_score dominates value_score by a huge
     ratio, the search is prior-driven (Q ignored) -> the "no within-position
     signal" is structural (flat normalized-Q has tiny *absolute* effect even
     after [0,1] normalization because the visited children all cluster).
  2. Re-run WITH Dirichlet noise (matching generation eps/alpha) and recompute
     the stored-vs-recomputed policy diff -> does noise explain anomaly B?
  3. Validate the stored policy target: sums to 1? supported only on legal moves?
  4. Inspect whether the MinMaxStats normalization is actually applied (span>0)
     and how many DISTINCT children ever get visited (visit concentration).
"""
import argparse
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory, stack_with_history
from src.mcts.mcts import MCTS, MinMaxStats, select_action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint_health import build_network


def load_games(buf_path, game, max_games=300):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/chess/2026_06_19_cold2_pc")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n-positions", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
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

    games = load_games(buf_path, game, max_games=300)
    sp = [g for g in games if not g.external_values and len(g.actions) >= 10]
    rng = np.random.default_rng(args.seed); rng.shuffle(sp)

    samples = []
    for g in sp:
        L = len(g.actions)
        ply = int(rng.integers(2, L - 1))
        if ply < len(g.policies):
            samples.append((g, ply))
        if len(samples) >= args.n_positions:
            break

    print(f"sims={args.sims} positions={len(samples)} eps={cfg.dirichlet_epsilon} alpha={cfg.dirichlet_alpha}", flush=True)
    print(f"{'ply':>4s} {'nleg':>4s} {'#vis>0':>6s} {'topVisFr':>8s} "
          f"{'mmSpan':>7s} {'medPriS':>8s} {'medValS':>8s} {'val/pri':>8s} "
          f"{'B_noise':>8s} {'B_clean':>8s} {'spVisPri':>8s}", flush=True)
    print("-" * 110, flush=True)

    agg = dict(valpri=[], bnoise=[], bclean=[], illeg=[], nvis=[], topfr=[], span=[])
    for (g, ply) in samples:
        cur = g.observations[ply]; prior = g.observations[:ply]
        obs = stack_with_history(cur, prior, HF)
        legal = g.legal_actions_list[ply]

        # --- Clean run (no noise) ---
        root = mcts.run(obs, legal, add_noise=False)
        visits = root.child_visits; actions = root.child_actions
        priors = root.child_priors
        rew = root.child_rewards; vs = root.child_value_sums
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.where(visits > 0, vs / np.maximum(visits, 1.0), 0.0)
        q = rew - cfg.discount * cv

        # Reconstruct FINAL-root PUCT decomposition (node.visit_count == root.visit_count).
        Nroot = root.visit_count
        pb_c = math.log((Nroot + MCTS.PB_C_BASE + 1) / MCTS.PB_C_BASE) + MCTS.PB_C_INIT
        sqrt_n = math.sqrt(Nroot)
        prior_score = pb_c * priors * sqrt_n / (1.0 + visits)
        # MinMaxStats final span: recompute from visited children's mover-Q (proxy
        # for the actual mm seen; the true mm also saw intermediate nodes but the
        # root children dominate the range here).
        vmask = visits > 0
        if vmask.sum() >= 2:
            mm_min = q[vmask].min(); mm_max = q[vmask].max()
        else:
            mm_min = mm_max = 0.0
        span = mm_max - mm_min
        if span > 0:
            nq = (q - mm_min) / span
        else:
            nq = q
        value_score = np.where(vmask, nq, 0.0)

        med_pri_s = float(np.median(prior_score[vmask])) if vmask.any() else float("nan")
        med_val_s = float(np.median(value_score[vmask])) if vmask.any() else float("nan")
        valpri = med_val_s / med_pri_s if med_pri_s > 1e-12 else float("nan")

        vsum = visits.sum()
        topfr = float(visits.max() / vsum) if vsum > 0 else float("nan")
        nvis = int(vmask.sum())

        # --- Stored policy validity ---
        stored = g.policies[ply].astype(np.float64)
        pol_sum = float(stored.sum())
        legal_set = set(int(a) for a in legal)
        illeg_mass = float(sum(stored[a] for a in range(len(stored)) if a not in legal_set))

        # --- B clean: stored vs recomputed (no noise) ---
        _, rec_clean = select_action(root, temperature=1.0)
        m = min(len(rec_clean), len(stored))
        b_clean = float(np.abs(rec_clean[:m] - stored[:m]).max())

        # --- B noise: stored vs recomputed WITH Dirichlet noise (gen-matched) ---
        rn = mcts.run(obs, legal, add_noise=True)
        _, rec_n = select_action(rn, temperature=1.0)
        mm = min(len(rec_n), len(stored))
        b_noise = float(np.abs(rec_n[:mm] - stored[:mm]).max())

        # --- KEY: do visits track PRIOR or Q? spearman(visits, prior) over visited ---
        from probe_mcts_internal_consistency import spearman
        sp_vp = spearman(visits[vmask], priors[vmask]) if vmask.sum() >= 2 else float("nan")
        sp_vq = spearman(visits[vmask], q[vmask]) if vmask.sum() >= 2 else float("nan")

        print(f"{ply:4d} {len(actions):4d} {nvis:6d} {topfr:8.3f} "
              f"{span:7.3f} {med_pri_s:8.4f} {med_val_s:8.4f} {valpri:8.2f} "
              f"{b_noise:8.4f} {b_clean:8.4f} vp={sp_vp:+.2f} vq={sp_vq:+.2f}", flush=True)
        agg["valpri"].append(valpri); agg["bnoise"].append(b_noise)
        agg["bclean"].append(b_clean); agg["illeg"].append(illeg_mass)
        agg["nvis"].append(nvis); agg["topfr"].append(topfr); agg["span"].append(span)
        agg.setdefault("spvp", []).append(sp_vp)
        agg.setdefault("spvq", []).append(sp_vq)

    print("\nAGGREGATE")
    def s(name, key, fmt="{:.4f}"):
        a = np.array([x for x in agg[key] if not np.isnan(x)], dtype=np.float64)
        if len(a) == 0:
            print(f"  {name:30s} (no data)"); return
        print(f"  {name:30s} mean={fmt.format(a.mean())} med={fmt.format(np.median(a))} "
              f"min={fmt.format(a.min())} max={fmt.format(a.max())}")
    s("val/pri score ratio", "valpri", "{:.3f}")
    s("B stored-vs-recomputed (NOISE)", "bnoise")
    s("B stored-vs-recomputed (clean)", "bclean")
    s("illegal mass in stored policy", "illeg", "{:.6f}")
    s("# children visited (>0)", "nvis", "{:.1f}")
    s("top child visit fraction", "topfr", "{:.3f}")
    s("MinMaxStats span (visited Q)", "span")
    s("spearman(visits, PRIOR)", "spvp", "{:+.3f}")
    s("spearman(visits, Q)", "spvq", "{:+.3f}")
    print("\n  val/pri << 1 => PUCT prior-dominated (search ignores Q).")
    print("  B(noise) ~ B(clean): noise explains little; B(noise) << B(clean): noise is the cause.")
    print("  spearman(visits,PRIOR) >> spearman(visits,Q) => visits track prior, NOT Q (flat-Q symptom).")


if __name__ == "__main__":
    main()
