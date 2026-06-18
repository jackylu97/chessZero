"""V^pi calibration: is the model's value honestly calibrated to its OWN self-play
outcomes? (The correct health metric for a self-play value — no Stockfish involved.)

Bins self-play positions by the model's predicted root value V, and reports the
ACTUAL self-play outcome (STM POV) in each bin. A well-calibrated value tracks the
diagonal (E[outcome | V=v] ~ v). A value that over-predicts wins that don't
materialize shows a FLAT curve (high V, but outcomes ~0 because the games draw) —
the draw-basin signature on the value's own terms.

Run: .venv/bin/python scripts/probe_value_calibration.py --buf <path.buf>
"""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--min-ply", type=int, default=8, help="skip random-opening plies")
    args = ap.parse_args()

    recs = []
    with open(args.buf, "rb") as f:
        header = pickle.load(f)
        for _ in range(header["n_records"]):
            d, _prio = pickle.load(f)
            recs.append(d)
    sp = [d for d in recs if len(d.get("external_values", [])) == 0]
    print(f"{args.buf}: {len(recs)} games ({len(sp)} self-play)\n")

    pred, actual = [], []
    for d in sp:
        z = float(d["game_outcome"])
        rv = d.get("root_values", [])
        n = len(d["actions"])
        for ply in range(args.min_ply, n):
            if ply >= len(rv):
                break
            v = float(rv[ply])
            stm_white = (ply % 2 == 0)
            pred.append(v)
            actual.append(z if stm_white else -z)
    pred = np.array(pred); actual = np.array(actual)
    print(f"positions: {len(pred)} (self-play, ply>={args.min_ply})")
    print(f"overall: corr(V, outcome)={np.corrcoef(pred, actual)[0,1]:+.3f}  "
          f"mean|V|={np.abs(pred).mean():.3f}  mean|outcome|={np.abs(actual).mean():.3f}  "
          f"(decisive-game frac {np.mean(actual!=0):.0%})\n")

    edges = [-1.01, -0.6, -0.35, -0.15, -0.05, 0.05, 0.15, 0.35, 0.6, 1.01]
    print(f"{'predicted V bin':>16} {'n':>6} {'mean V':>8} {'mean outcome':>13} "
          f"{'%win':>6} {'%draw':>6} {'%loss':>6}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred > lo) & (pred <= hi)
        if m.sum() == 0:
            continue
        a = actual[m]
        print(f"{f'({lo:+.2f},{hi:+.2f}]':>16} {int(m.sum()):>6} {pred[m].mean():>+8.3f} "
              f"{a.mean():>+13.3f} {np.mean(a>0):>6.0%} {np.mean(a==0):>6.0%} {np.mean(a<0):>6.0%}")

    # The key draw-basin test: of positions the value calls clearly winning, how
    # many actually win vs draw?
    for thr in (0.3, 0.5):
        m = pred > thr
        if m.sum():
            a = actual[m]
            print(f"\n  V > +{thr}: n={int(m.sum())}  actually win {np.mean(a>0):.0%}, "
                  f"draw {np.mean(a==0):.0%}, lose {np.mean(a<0):.0%}  (mean outcome {a.mean():+.2f})")
    print("\n  Calibrated if 'mean outcome' tracks 'mean V' down the table.")
    print("  Draw-basin-on-its-own-terms if high-V bins still mostly DRAW (mean outcome << V).")


if __name__ == "__main__":
    main()
