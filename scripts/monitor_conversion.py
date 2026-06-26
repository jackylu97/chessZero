"""Conversion monitor: are SEEDED endgames and SELF-REACHED endgames converted at the
same rate? If seeded converts but self-reached doesn't, the manifolds are disjoint and
the seed sampler should be reweighted toward the model's own reached-endgame histogram.

For each self-play game we replay it and ask:
  - 'had a winning endgame': some ply has <=5 pieces with Syzygy wdl==2 for the side to
    move (or, for SEEDED games, the seed start position itself). Record the winning side.
  - 'converted': the game ended in a REAL checkmate (board.is_checkmate) delivered BY the
    winning side. Resignation-relabels end decisive but NOT in checkmate, so they're
    excluded — only model-delivered mates count.
Partition by seeded (GameHistory.start_fen set) vs self-reached, report conversion rate.

Run: PYTHONPATH=. .venv/bin/python scripts/monitor_conversion.py --buf <ckpt.buf>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess, chess.syzygy
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer


def analyze(game_hist, tb, max_pieces=5):
    """Returns (seeded, had_winning, converted) for one game, or None if unusable."""
    seeded = bool(getattr(game_hist, "start_fen", None))
    b = chess.Board(getattr(game_hist, "start_fen", None) or chess.STARTING_FEN)
    boards = [b.copy()]
    for a in game_hist.actions:
        mv = _action_to_move(int(a), b)
        if mv is None or mv not in b.legal_moves:
            break
        b.push(mv); boards.append(b.copy())
    win_side = None
    for bd in boards[:-1]:           # winning side at the LAST winning <=5pc ply
        if len(bd.piece_map()) <= max_pieces:
            try:
                if tb.probe_wdl(bd) == 2:
                    win_side = bd.turn
            except Exception:
                pass
    if win_side is None:
        return seeded, False, False
    final = boards[-1]
    converted = final.is_checkmate() and (not final.turn) == win_side
    return seeded, True, converted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--max-games", type=int, default=4000)
    ap.add_argument("--recent", action="store_true",
                    help="Analyze the FRESHEST max-games (buffer TAIL) instead of the oldest "
                         "(head). Required for trend analysis: the head self-play games are the "
                         "same oldest games shared across every later checkpoint, so the head "
                         "rate looks frozen. Use --recent to measure games near this checkpoint.")
    args = ap.parse_args()
    game = ChessGame()
    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    _sp_all = [g for g in rb.buffer if not getattr(g, "external_values", [])]
    sp = _sp_all[-args.max_games:] if args.recent else _sp_all[: args.max_games]
    tb = chess.syzygy.open_tablebase(args.tb)

    stats = {"seeded": [0, 0], "self": [0, 0]}   # key -> [had_winning, converted]
    for g in sp:
        seeded, had, conv = analyze(g, tb)
        if not had:
            continue
        k = "seeded" if seeded else "self"
        stats[k][0] += 1
        stats[k][1] += int(conv)
    tb.close()

    print(f"{args.buf}: {len(sp)} self-play games\n")
    print(f"{'partition':>10} {'had_winning':>12} {'converted':>10} {'conv_rate':>10}")
    for k in ("seeded", "self"):
        had, conv = stats[k]
        rate = conv / had if had else float("nan")
        print(f"{k:>10} {had:>12} {conv:>10} {rate:>9.1%}")
    sh, ss = stats["seeded"], stats["self"]
    if sh[0] and ss[0]:
        gap = (sh[1]/sh[0]) - (ss[1]/ss[0])
        print(f"\n  seeded - self conversion gap: {gap:+.1%}")
        print("  large positive gap => seeded converts but self-reached doesn't => "
              "reweight seed sampler toward the model's own reached-endgame distribution.")
    elif not ss[0]:
        print("\n  (no self-reached winning endgames in buffer)")
    elif not sh[0]:
        print("\n  (no seeded games in buffer — run a seeded arm to populate)")


if __name__ == "__main__":
    main()
