"""Efficacy dashboard for the DTZ value-target relabeling.

Runs the leading indicators on a checkpoint and scores each against the measured
PRE-relabel baseline (run 2026_06_25_tb_probe @ ~54k):
  - value-head DTZ correlation   baseline -0.34 (INVERTED)  -> PASS if > 0
  - net prior DTZ-optimality     baseline 70.9%             -> PASS if > 90%
  - no-probe KQvK conversion     baseline hangs queen/draw  -> PASS if MATE
  - no-probe KRvK conversion     baseline shuffles to cap   -> PASS if MATE

All probes run on CPU by default so they don't contend with a live GPU training
run. Wraps the existing probe scripts (single source of truth).

Run: PYTHONPATH=. .venv/bin/python scripts/eval_relabel_dashboard.py \
        --checkpoint checkpoints/chess/2026_06_25_tb_value/checkpoint_70000.pt
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (label, baseline_str, pass_predicate(value) -> bool)
BASELINE = {
    "value_dtz_corr":  ("-0.34 (inverted)", lambda v: v is not None and v > 0.0),
    "prior_dtz_opt":   ("70.9%",            lambda v: v is not None and v > 90.0),
    "KQvK_no_probe":   ("hangs queen/draw", lambda v: v == "MATE"),
    "KRvK_no_probe":   ("shuffles to cap",  lambda v: v == "MATE"),
}


def run(cmd):
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return p.stdout + "\n" + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--buf", default=None, help="default: sibling .buf of the checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--max-plies", type=int, default=140)
    ap.add_argument("--skip-corr", action="store_true", help="skip the (slow) buffer net-eval")
    args = ap.parse_args()

    py = os.path.join(ROOT, ".venv/bin/python")
    buf = args.buf or (args.checkpoint[:-3] + ".buf")
    step = re.search(r"checkpoint_(\d+)", args.checkpoint)
    step = step.group(1) if step else "?"
    results = {}

    # --- value-head DTZ correlation + prior DTZ-optimality (buffer net-eval) ---
    if not args.skip_corr and os.path.exists(buf):
        out = run([py, "scripts/probe_net_tb_eval.py", "--buf", buf,
                   "--checkpoint", args.checkpoint, "--device", args.device,
                   "--max-games", str(args.max_games)])
        m = re.search(r"corr\(net child value, -DTZ\):\s+mean\s+([+-][\d.]+)", out)
        results["value_dtz_corr"] = float(m.group(1)) if m else None
        m = re.search(r"net prior argmax is DTZ-OPTIMAL:\s+([\d.]+)%", out)
        results["prior_dtz_opt"] = float(m.group(1)) if m else None
    else:
        results["value_dtz_corr"] = None
        results["prior_dtz_opt"] = None
        if not os.path.exists(buf):
            print(f"(no buffer at {buf} — skipping corr/prior metrics)")

    # --- no-probe endgame conversion (KQvK / KRvK) ---
    out = run([py, "scripts/probe_tb_conversion.py", "--checkpoint", args.checkpoint,
               "--device", args.device, "--sims", str(args.sims), "--no-probe",
               "--dtz-weights", "1.0", "--max-plies", str(args.max_plies)])
    for name in ("KQvK", "KRvK"):
        m = re.search(rf"^\s*{name}\s+[\d.]+\s+(\S+)\s+(\d+)\s+(\d+)", out, re.M)
        if m:
            results[f"{name}_no_probe"] = m.group(1)
            results[f"{name}_plies"] = int(m.group(2))
        else:
            results[f"{name}_no_probe"] = "?"
            results[f"{name}_plies"] = None

    # --- report ---
    print(f"\n{'='*72}\n  RELABEL EFFICACY DASHBOARD — step {step}\n"
          f"  {os.path.basename(args.checkpoint)}\n{'='*72}")
    print(f"  {'metric':<20}{'baseline':<22}{'current':<16}{'verdict'}")
    print(f"  {'-'*68}")
    def row(key, cur_str, cur_val):
        base, pred = BASELINE[key]
        verdict = "PASS ✓" if pred(cur_val) else "----"
        print(f"  {key:<20}{base:<22}{cur_str:<16}{verdict}")
    cv = results["value_dtz_corr"]
    row("value_dtz_corr", "n/a" if cv is None else f"{cv:+.3f}", cv)
    pv = results["prior_dtz_opt"]
    row("prior_dtz_opt", "n/a" if pv is None else f"{pv:.1f}%", pv)
    row("KQvK_no_probe", f'{results["KQvK_no_probe"]} ({results["KQvK_plies"]}p)',
        results["KQvK_no_probe"])
    row("KRvK_no_probe", f'{results["KRvK_no_probe"]} ({results["KRvK_plies"]}p)',
        results["KRvK_no_probe"])
    passed = sum(1 for k in BASELINE if BASELINE[k][1](results.get(k)))
    print(f"  {'-'*68}\n  {passed}/4 indicators improved over baseline.\n")


if __name__ == "__main__":
    main()
