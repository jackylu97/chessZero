"""Trajectory eval harness — the primary self-play-improvement screen.

Sweeps a run's checkpoints and reads the SIBLING-RANKING trajectory: for each
checkpoint it runs scripts/probe_sibling_ranking.py (within-position Spearman of
the value head vs Stockfish over a node's legal moves — the confirmed-broken
quantity, see memory value-sibling-ranking). The probe is CPU + Stockfish, so the
sweep does NOT contend with the training GPU and many checkpoints run in parallel.

The signal is a TRAJECTORY: is within-position Spearman / best-move-agreement
RISING over training steps (and faster for an ablation arm than the baseline)?
That is sensitive long before head-to-head Elo moves at short horizons.

Run:
  .venv/bin/python scripts/eval_trajectory.py --run-id 2026_06_19_cold_posctrl \
      --game chess_small --positions 24 --sims 60 --workers 8

Note: chess_small checkpoints are saved on disk under checkpoints/chess/<run-id>/
(config.game == "chess"), so --ckpt-game-dir defaults to "chess" while --game
(network sizing) is chess_small. Override --ckpt-game-dir for other games.
"""
import argparse, os, sys, json, glob, re, subprocess, csv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def discover(ckpt_dir, steps_filter):
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")):
        m = re.search(r"checkpoint_(\d+)\.pt$", p)
        if not m:
            continue
        s = int(m.group(1))
        if steps_filter and s not in steps_filter:
            continue
        out.append((s, p))
    return sorted(out)


def run_probe(path, game, positions, sims, sf_depth, stockfish, seed):
    cmd = [os.path.join(ROOT, ".venv/bin/python"),
           os.path.join(ROOT, "scripts/probe_sibling_ranking.py"),
           "--checkpoint", path, "--game", game, "--json",
           "--positions", str(positions), "--sims", str(sims),
           "--sf-depth", str(sf_depth), "--stockfish", stockfish, "--seed", str(seed)]
    # Pin each probe to 1 thread: torch/numpy/BLAS otherwise each grab all cores, so
    # N parallel probes oversubscribe and run slower than serial. 1 thread/probe →
    # the ThreadPool's N workers map cleanly to N cores.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                           cwd=ROOT, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if r.returncode != 0:
        return {"error": f"rc={r.returncode}: {r.stderr.strip()[-200:]}"}
    for line in reversed([l for l in r.stdout.strip().splitlines() if l.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"error": f"no JSON in output: {r.stdout.strip()[-200:]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="run id under checkpoints/<ckpt-game-dir>/")
    ap.add_argument("--checkpoint-dir", help="explicit checkpoint dir (overrides --run-id)")
    ap.add_argument("--game", default="chess_small", help="config preset for network sizing")
    ap.add_argument("--ckpt-game-dir", default="chess",
                    help="on-disk game dir (chess_small saves under 'chess').")
    ap.add_argument("--steps", default="", help="comma list of steps to include (default: all)")
    ap.add_argument("--positions", type=int, default=24)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="", help="CSV output path (default eval_traj_<run>.csv)")
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir or os.path.join(
        ROOT, "checkpoints", args.ckpt_game_dir, args.run_id or "")
    steps_filter = {int(x) for x in args.steps.split(",") if x.strip()} if args.steps else None
    ckpts = discover(ckpt_dir, steps_filter)
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        sys.exit(1)
    print(f"Sweeping {len(ckpts)} checkpoints in {ckpt_dir}")
    print(f"  positions={args.positions} sims={args.sims} sf_depth={args.sf_depth} "
          f"workers={args.workers}\n")

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_probe, p, args.game, args.positions, args.sims,
                          args.sf_depth, args.stockfish, args.seed): s for s, p in ckpts}
        for fut in as_completed(futs):
            s = futs[fut]
            results[s] = fut.result()
            r = results[s]
            if "error" in r:
                print(f"  step {s:>7}: ERROR {r['error']}", flush=True)
            else:
                print(f"  step {s:>7}: within-median {r['within_spearman_median']:+.3f}  "
                      f"best-agree {r['best_move_agreement']:.0%}  "
                      f"pooled(wide) {r['pooled_wide_spearman']:+.2f}  n={r['n_positions']}",
                      flush=True)

    print(f"\n{'step':>8} {'within_med':>11} {'within_mean':>12} {'best_agree':>11} "
          f"{'sf_rank_med':>12} {'pooled_wide':>12}")
    print("  " + "-" * 70)
    ok_rows = []
    for s, _ in ckpts:
        r = results.get(s, {})
        if "error" in r:
            print(f"{s:>8}   (error: {r['error']})")
            continue
        ok_rows.append(r)
        print(f"{s:>8} {r['within_spearman_median']:>+11.3f} {r['within_spearman_mean']:>+12.3f} "
              f"{r['best_move_agreement']:>10.0%} {r['model_move_sf_rank_median']:>12.1f} "
              f"{r['pooled_wide_spearman']:>+12.2f}")

    out = args.out or os.path.join(ROOT, f"eval_traj_{args.run_id or 'run'}.csv")
    if ok_rows:
        keys = list(ok_rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(ok_rows)
        print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
