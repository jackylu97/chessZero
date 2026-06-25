"""Are the POLICY/VALUE TARGETS at winning TB roots actually teaching conversion?

The conversion demos exist in the buffer (17.5% of games mate from <=5 pieces).
So if the model still can't convert without the probe, the suspect is the TARGET
the model is trained on at those positions. For every replayed root that is a
CLEAN Syzygy win for the side to move (wdl==2), we score the STORED policy target
(the exact visit distribution used as the training label, already shaped by the
soft TB value bias) against Syzygy ground truth:

  - pmax            : max prob in the target (sharp vs flat)
  - p_preserve      : policy mass on win-PRESERVING moves (wdl_child < 0)
  - p_optimal       : policy mass on DTZ-OPTIMAL moves (fastest mate)
  - argmax_preserve : top-visit move keeps the win?
  - argmax_optimal  : top-visit move is DTZ-optimal?
  - root_value      : stored value target (decisive ~+1 vs washed ~0)

If pmax is low / p_optimal is low / root_value is washed, the model is trained to
imitate a near-random walk among winning moves -> never learns to MAKE PROGRESS,
even though the games eventually mate. That is the "missing MCTS mechanism".

Run: PYTHONPATH=. .venv/bin/python scripts/probe_tb_target_quality.py --buf <ckpt.buf>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch  # noqa
import chess, chess.syzygy

from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--max-pieces", type=int, default=5)
    ap.add_argument("--max-games", type=int, default=900)
    ap.add_argument("--max-roots", type=int, default=4000)
    args = ap.parse_args()

    game = ChessGame()
    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])][: args.max_games]
    tb = chess.syzygy.open_tablebase(args.tb)
    print(f"{args.buf}: {len(sp)} self-play games scanned\n")

    rows = []  # per winning-TB-root metrics
    for g in sp:
        actions = list(getattr(g, "actions", []) or [])
        policies = getattr(g, "policies", None)
        rvs = getattr(g, "root_values", None)
        if not actions or policies is None:
            continue
        b = chess.Board()
        sfen = getattr(g, "start_fen", None)
        if sfen:
            try: b = chess.Board(sfen)
            except Exception: b = chess.Board()
        for t, a in enumerate(actions):
            npc = len(b.piece_map())
            if npc <= args.max_pieces and t < len(policies):
                try:
                    wdl = tb.probe_wdl(b)
                except Exception:
                    wdl = None
                if wdl == 2:  # clean win for side to move
                    pol = policies[t]
                    idx, prob = (pol if isinstance(pol, tuple) else (np.asarray(list(pol.keys())),
                                                                     np.asarray(list(pol.values()))))
                    idx = np.asarray(idx); prob = np.asarray(prob, dtype=np.float64)
                    if prob.sum() > 0: prob = prob / prob.sum()
                    # classify each candidate move in the target
                    preserve_mass = optimal_mass = 0.0
                    best_dtz = None; move_dtz = {}; legal_keep = {}
                    for ai, pv in zip(idx, prob):
                        mv = _action_to_move(int(ai), b)
                        if mv is None or mv not in b.legal_moves:
                            continue
                        b.push(mv)
                        try:
                            keep = b.is_checkmate() or (tb.probe_wdl(b) < 0)
                            d = 0 if b.is_checkmate() else abs(tb.probe_dtz(b))
                        except Exception:
                            keep = False; d = None
                        b.pop()
                        if keep:
                            preserve_mass += pv
                            legal_keep[int(ai)] = pv
                            if d is not None:
                                move_dtz[int(ai)] = d
                                best_dtz = d if best_dtz is None else min(best_dtz, d)
                    if best_dtz is not None:
                        opt = {ai for ai, d in move_dtz.items() if d == best_dtz}
                        optimal_mass = sum(pv for ai, pv in zip(idx, prob) if int(ai) in opt)
                    # argmax-visit move
                    am = int(idx[int(np.argmax(prob))])
                    am_pres = am in legal_keep
                    am_opt = best_dtz is not None and am in {ai for ai, d in move_dtz.items() if d == best_dtz}
                    rv = float(rvs[t]) if rvs is not None and t < len(rvs) else float("nan")
                    rows.append((float(prob.max()), preserve_mass, optimal_mass,
                                 int(am_pres), int(am_opt), rv, npc))
            try:
                mv = _action_to_move(int(a), b)
                if mv is None or mv not in b.legal_moves:
                    break
                b.push(mv)
            except Exception:
                break
        if len(rows) >= args.max_roots:
            break
    tb.close()

    if not rows:
        print("no winning TB roots found"); return
    R = np.array(rows, dtype=np.float64)
    pmax, pres, opt, ampres, amopt, rv, npc = [R[:, i] for i in range(7)]
    finite_rv = rv[np.isfinite(rv)]
    print(f"=== {len(R)} clean-win (wdl=2) TB roots, <= {args.max_pieces} pieces ===")
    print(f"  policy pmax (sharpness):          mean {pmax.mean():.3f}  median {np.median(pmax):.3f}")
    print(f"  policy mass on WIN-PRESERVING:    mean {pres.mean():.3f}")
    print(f"  policy mass on DTZ-OPTIMAL:       mean {opt.mean():.3f}")
    print(f"  top-visit move PRESERVES win:     {ampres.mean():.1%}")
    print(f"  top-visit move is DTZ-OPTIMAL:    {amopt.mean():.1%}")
    print(f"  root_value target:                mean {finite_rv.mean():+.3f}  "
          f"median {np.median(finite_rv):+.3f}  (n={len(finite_rv)})")
    print(f"    frac root_value > +0.5:         {(finite_rv > 0.5).mean():.1%}")
    print(f"    frac root_value in [-0.2,0.2]:  {((finite_rv>-0.2)&(finite_rv<0.2)).mean():.1%}")
    print(f"\n  INTERPRETATION: high pmax + high DTZ-optimal mass + decisive root_value")
    print(f"  => good conversion signal. Flat pmax / low optimal mass / washed value")
    print(f"  => the target teaches a random walk among winning moves (no progress).")


if __name__ == "__main__":
    main()
