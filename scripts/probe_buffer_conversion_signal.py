"""How much actual conversion-training signal is in the buffer?

The TB probe only fires at <= tb_max_pieces, and resignation truncates most won
games before they're played out. So the games that actually carry a *played-out*
conversion are: reached <= N pieces AND not resigned (ended in a real mate). This
quantifies that fraction. Flags aren't serialized, so we reconstruct by replay:
  - probe-affected  = min piece count over the game <= tb_max_pieces
  - resigned        = decisive outcome but final board is NOT checkmate (truncated)
  - natural mate    = decisive outcome AND final board is checkmate

Run: .venv/bin/python scripts/probe_buffer_conversion_signal.py --buf <ckpt.buf> --max-pieces 5
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch  # noqa: F401 (unpickle observation tensors)
import chess

from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--max-pieces", type=int, default=5)
    ap.add_argument("--max-games", type=int, default=6000)
    args = ap.parse_args()

    game = ChessGame()
    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])][: args.max_games]
    print(f"buffer {args.buf}: {len(rb.buffer)} games, {len(sp)} self-play\n")

    n = 0
    reached = resigned = mate = draw = plycap = 0
    reached_and_played = reached_and_mate = reached_and_resigned = 0
    for g in sp:
        actions = list(getattr(g, "actions", []) or [])
        if not actions:
            continue
        n += 1
        s = game.reset(); minpc = 32; ok = True
        for a in actions:
            try:
                s, _, _ = game.step(s, a)
            except Exception:
                ok = False; break
            minpc = min(minpc, len(s.board.piece_map()))
        if not ok:
            n -= 1; continue
        final = s.board
        outcome = float(getattr(g, "game_outcome", 0.0))
        is_mate = final.is_checkmate()
        in_tb = minpc <= args.max_pieces
        is_resigned = (outcome != 0.0) and (not is_mate) and (not final.is_game_over())
        is_draw = (outcome == 0.0)

        if in_tb: reached += 1
        if is_resigned: resigned += 1
        elif is_mate: mate += 1
        elif is_draw: draw += 1
        else: plycap += 1  # decisive + game_over-but-not-mate (rare)

        if in_tb and not is_resigned: reached_and_played += 1
        if in_tb and is_mate: reached_and_mate += 1
        if in_tb and is_resigned: reached_and_resigned += 1

    p = lambda x: f"{x}/{n} = {x/max(1,n):.1%}"
    print(f"=== over {n} self-play games (tb_max_pieces={args.max_pieces}) ===")
    print(f"  reached <= {args.max_pieces} pieces (probe COULD fire):  {p(reached)}")
    print(f"  resigned (truncated decisive):                {p(resigned)}")
    print(f"  natural mate:                                 {p(mate)}")
    print(f"  draw:                                         {p(draw)}")
    print(f"  ply-capped/other:                             {p(plycap)}")
    print(f"\n  --- the conversion-signal fraction ---")
    print(f"  probe-affected AND not resigned (played out): {p(reached_and_played)}")
    print(f"  probe-affected AND ended in MATE (real demo): {p(reached_and_mate)}")
    print(f"  probe-affected BUT resigned (signal lost):    {p(reached_and_resigned)}")


if __name__ == "__main__":
    main()
