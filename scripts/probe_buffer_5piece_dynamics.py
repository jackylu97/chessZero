"""Buffer dynamics: does the model reach 5-piece positions and then immediately
simplify (probe-steered DTZ=1 captures), so it never accumulates 5-piece
maneuvering experience? Reports per-piece-count ply counts, dwell time (consecutive
plies at exactly that piece count), DTZ spread per piece count, and whether the
PLAYED move at 5-piece winning positions is the DTZ-optimal (probe-steered) move.

Run: PYTHONPATH=. .venv/bin/python scripts/probe_buffer_5piece_dynamics.py --buf <ckpt.buf>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess, chess.syzygy
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--max-games", type=int, default=2500)
    args = ap.parse_args()
    game = ChessGame()
    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])][: args.max_games]
    tb = chess.syzygy.open_tablebase(args.tb)

    ply_count = {}                          # piece_count -> #plies
    dtz_by_pc = {3: [], 4: [], 5: []}       # winning-position DTZ per piece count
    dwell = {3: [], 4: [], 5: []}           # consecutive plies at exactly pc (per visit)
    steer_hit = steer_tot = 0               # 5-piece: played move DTZ-optimal?
    for g in sp:
        b = chess.Board(getattr(g, "start_fen", None) or chess.STARTING_FEN)
        run_pc, run_len = None, 0
        for ply, a in enumerate(g.actions):
            npc = len(b.piece_map())
            ply_count[npc] = ply_count.get(npc, 0) + 1
            if npc != run_pc:
                if run_pc in dwell:
                    dwell[run_pc].append(run_len)
                run_pc, run_len = npc, 1
            else:
                run_len += 1
            if npc in dtz_by_pc:
                try:
                    if tb.probe_wdl(b) == 2:
                        d = abs(int(tb.probe_dtz(b)))
                        dtz_by_pc[npc].append(d)
                        if npc == 5:  # is the PLAYED move DTZ-optimal? (probe-steered)
                            best = None; played_d = None
                            for mv in b.legal_moves:
                                b.push(mv)
                                try:
                                    k = b.is_checkmate() or (tb.probe_wdl(b) < 0)
                                    dd = 0 if b.is_checkmate() else abs(int(tb.probe_dtz(b)))
                                except Exception:
                                    k = False; dd = None
                                b.pop()
                                if k and dd is not None:
                                    best = dd if best is None else min(best, dd)
                            pm = _action_to_move(int(a), b)
                            if pm is not None and pm in b.legal_moves:
                                b.push(pm)
                                try:
                                    played_d = 0 if b.is_checkmate() else abs(int(tb.probe_dtz(b)))
                                    pk = b.is_checkmate() or (tb.probe_wdl(b) < 0)
                                except Exception:
                                    played_d = None; pk = False
                                b.pop()
                                if pk and played_d is not None and best is not None:
                                    steer_tot += 1
                                    if played_d == best:
                                        steer_hit += 1
                except Exception:
                    pass
            mv = _action_to_move(int(a), b)
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv)
        if run_pc in dwell:
            dwell[run_pc].append(run_len)
    tb.close()

    print(f"{len(sp)} self-play games\n")
    print("=== plies spent at each piece count (TB range) ===")
    for pc in sorted([k for k in ply_count if k <= 7]):
        print(f"  {pc} pieces: {ply_count[pc]:>7} plies")
    print("\n=== winning-position DTZ spread per piece count ===")
    for pc in (5, 4, 3):
        a = np.array(dtz_by_pc[pc])
        if len(a):
            print(f"  {pc}pc: n={len(a):>6}  DTZ min {a.min()} max {a.max()} mean {a.mean():.1f} "
                  f"| frac DTZ==1: {(a==1).mean():.0%}")
        else:
            print(f"  {pc}pc: none")
    print("\n=== dwell: consecutive plies at exactly that piece count (per visit) ===")
    for pc in (5, 4, 3):
        d = np.array(dwell[pc])
        if len(d):
            print(f"  {pc}pc: visits={len(d)}  mean dwell {d.mean():.1f} plies  median {np.median(d):.0f}  max {d.max()}")
    print(f"\n=== probe-steering at 5-piece winning positions ===")
    if steer_tot:
        print(f"  played move is DTZ-OPTIMAL: {steer_hit}/{steer_tot} = {steer_hit/steer_tot:.1%} "
              f"(high => self-play 5-piece play is probe-steered, not learned)")


if __name__ == "__main__":
    main()
