"""Inspect decisive self-play games in the buffer: are their policy/value targets
actually teaching conversion? Tests the hypothesis that decisive games are SHORT
tactical wins (won early) while the hard endgame conversions end up DRAWN (long) —
so the buffer over-represents wins the model can already get and under-represents
the skill it lacks.

Run: .venv/bin/python scripts/probe_decisive_targets.py --buf <path.buf>
"""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def decode_policies(d):
    """Return list of (entropy, n_nonzero) per step from the compact policy storage."""
    out = []
    if d["policies_mode"] == "onehot":
        for _ in d["policies_data"]:
            out.append((0.0, 1))
    else:  # sparse: list of (nz_indices, values)
        for item in d["policies_data"]:
            nz, vals = item
            v = np.asarray(vals, dtype=np.float64)
            v = v[v > 0]
            ent = float(-(v * np.log(v)).sum()) if len(v) else 0.0
            out.append((ent, len(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--min-ply", type=int, default=8)
    args = ap.parse_args()
    recs = []
    with open(args.buf, "rb") as f:
        header = pickle.load(f)
        for _ in range(header["n_records"]):
            d, _ = pickle.load(f)
            recs.append(d)
    sp = [d for d in recs if len(d.get("external_values", [])) == 0]
    dec = [d for d in sp if float(d["game_outcome"]) != 0.0]
    draw = [d for d in sp if float(d["game_outcome"]) == 0.0]
    print(f"{args.buf}: {len(sp)} self-play | decisive {len(dec)} ({len(dec)/max(1,len(sp)):.0%}) "
          f"| draw {len(draw)} ({len(draw)/max(1,len(sp)):.0%})\n")

    def lens(games): return np.array([len(g["actions"]) for g in games])
    dl, drl = lens(dec), lens(draw)
    print(f"game length:  decisive median {np.median(dl):.0f} mean {dl.mean():.0f} | "
          f"draw median {np.median(drl):.0f} mean {drl.mean():.0f}")

    # Win-clarity timing in decisive games: ply where the eventual winner's value
    # first crosses +0.5, as a fraction of game length. Early = tactical; late = endgame.
    fracs = []
    for d in dec:
        z = float(d["game_outcome"]); rv = d.get("root_values", [])
        n = len(d["actions"])
        cross = None
        for ply in range(n):
            if ply >= len(rv): break
            stm_white = (ply % 2 == 0)
            v_winner = rv[ply] if (stm_white == (z > 0)) else -rv[ply]  # winner-POV value at this ply
            if v_winner > 0.5:
                cross = ply / max(1, n); break
        if cross is not None:
            fracs.append(cross)
    fracs = np.array(fracs)
    print(f"\ndecisive win-clarity: winner's V first crosses +0.5 at game-fraction "
          f"median {np.median(fracs):.2f} mean {fracs.mean():.2f}  (n={len(fracs)})")
    print(f"  -> won in first half: {np.mean(fracs<0.5):.0%}  | last quarter: {np.mean(fracs>0.75):.0%}")

    # Policy target entropy (sharpness): decisive vs draw, and decisive late-game.
    def mean_ent(games, frac_lo=0.0, frac_hi=1.0):
        es = []
        for g in games:
            pe = decode_policies(g); n = len(pe)
            for i, (ent, _) in enumerate(pe):
                if i < args.min_ply: continue
                fr = i / max(1, n)
                if frac_lo <= fr < frac_hi:
                    es.append(ent)
        return np.mean(es) if es else float("nan")
    print(f"\npolicy-target entropy (nats; lower=sharper):")
    print(f"  decisive games (all)     {mean_ent(dec):.3f}")
    print(f"  decisive last 25% plies  {mean_ent(dec,0.75,1.0):.3f}   <- the conversion phase")
    print(f"  draw games (all)         {mean_ent(draw):.3f}")
    print(f"  draw last 25% plies      {mean_ent(draw,0.75,1.0):.3f}")

    # Sample one decisive game: value-target proxy + policy entropy over the tail.
    g = max(dec, key=lambda d: len(d["actions"]))  # a long decisive game (endgame conversion if any)
    z = float(g["game_outcome"]); rv = g["root_values"]; pe = decode_policies(g); n = len(g["actions"])
    print(f"\nsample longest decisive game (len {n}, outcome {z:+.0f}):  ply | winner-V | policy-entropy | #moves")
    for ply in list(range(max(args.min_ply, n-12), n)):
        if ply >= len(rv): break
        stm_white = (ply % 2 == 0)
        vw = rv[ply] if (stm_white == (z > 0)) else -rv[ply]
        ent, nz = pe[ply] if ply < len(pe) else (float('nan'), 0)
        print(f"    {ply:>4} {vw:>+9.2f} {ent:>15.2f} {nz:>8}")


if __name__ == "__main__":
    main()
