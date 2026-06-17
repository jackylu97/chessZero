"""Stockfish-audit the tail positions of self-play threefold-draw games:
are they objectively WON (model failing to convert) or GENUINELY drawn?

Run: .venv/bin/python scripts/audit_threefold_sf.py --buf <path.buf> --games 20
"""
import argparse, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import chess, chess.engine
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--tail", type=int, default=10, help="last N plies per game")
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    c = get_config("chess"); game = ChessGame()
    rb = ReplayBuffer(c); rb.load(args.buf, game=game)
    tf = [g for g in rb.buffer if not g.external_values and g.draw_by_repetition and g.game_outcome == 0]
    random.shuffle(tf)
    tf = tf[:args.games]
    print(f"{args.buf}: auditing tail-{args.tail} positions of {len(tf)} threefold self-play games, "
          f"SF depth {args.sf_depth}\n")

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    abs_best = []  # |eval| from the POV of whichever side is better (=how decisive)
    per_game = []
    for gi, gh in enumerate(tf):
        board = chess.Board(); ok = True
        for a in gh.actions:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                ok = False; break
            board.push(mv)
        if not ok:
            continue
        # walk back `tail` plies, SF-eval each
        # reconstruct the sequence of tail boards by replaying again and snapshotting
        b2 = chess.Board(); fens = []
        for a in gh.actions:
            mv = _action_to_move(int(a), b2)
            b2.push(mv); fens.append(b2.fen())
        tail_fens = fens[-args.tail:]
        evals = []
        for fen in tail_fens:
            bb = chess.Board(fen)
            if bb.is_game_over():
                continue
            info = eng.analyse(bb, chess.engine.Limit(depth=args.sf_depth))
            cp = info["score"].white().score(mate_score=10000) / 100.0  # white POV
            evals.append(cp)
        if not evals:
            continue
        ev = np.array(evals)
        mab = np.max(np.abs(ev))  # most-decisive tail position
        abs_best.append(mab)
        per_game.append((len(gh.actions), float(np.mean(ev)), float(mab)))

    eng.quit()
    ab = np.array(abs_best)
    print(f"{'game':>5} {'plies':>5} {'mean tail eval (W)':>18} {'max|eval| in tail':>18}")
    for i, (n, me, mab) in enumerate(per_game):
        print(f"{i:>5} {n:>5} {me:>+18.2f} {mab:>18.2f}")
    print(f"\n  Across {len(ab)} games, max|eval| in the threefold tail (pawns):")
    print(f"    median {np.median(ab):.2f} | mean {ab.mean():.2f}")
    print(f"    genuinely drawn (<0.5):   {np.mean(ab < 0.5):.0%}")
    print(f"    small edge (0.5-1.5):     {np.mean((ab>=0.5)&(ab<1.5)):.0%}")
    print(f"    winning (1.5-3):          {np.mean((ab>=1.5)&(ab<3)):.0%}")
    print(f"    decisive (>3):            {np.mean(ab>=3):.0%}")
    print("\n  -> high decisive% = objectively-won positions shuffled to draws (model blind/can't convert)")
    print("     high drawn%    = genuine draws (penalty would be miscalibrating them)")


if __name__ == "__main__":
    main()
