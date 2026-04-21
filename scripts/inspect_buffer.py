"""Inspect a saved replay buffer: game outcomes, lengths, capture behavior, sample moves.

Usage:
    python scripts/inspect_buffer.py --buffer checkpoints/chess/<run-id>/checkpoint_11000.buf
    python scripts/inspect_buffer.py --buffer <path> --num-games 10 --summary-only
"""

import argparse
import os
import pickle
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from src.games.chess import _action_to_move


def replay_game(game):
    """Decode a GameHistory's actions back into a chess.Board sequence.

    Returns (san_moves, num_captures, num_checks, terminated_cleanly).
    """
    board = chess.Board()
    san_moves = []
    captures = 0
    checks = 0
    for act in game.actions:
        move = _action_to_move(act, board)
        if move is None or move not in board.legal_moves:
            san_moves.append(f"<illegal:{act}>")
            return san_moves, captures, checks, False
        if board.is_capture(move):
            captures += 1
        if board.gives_check(move):
            checks += 1
        san_moves.append(board.san(move))
        board.push(move)
    return san_moves, captures, checks, True


def format_pgn_line(san_moves):
    pairs = []
    for i in range(0, len(san_moves), 2):
        w = san_moves[i]
        b = san_moves[i + 1] if i + 1 < len(san_moves) else ""
        pairs.append(f"{i // 2 + 1}. {w} {b}".strip())
    return " ".join(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--num-games", type=int, default=3,
                        help="Number of sample games to print in full")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    with open(args.buffer, "rb") as f:
        data = pickle.load(f)
    buf = data["buffer"]

    print(f"Buffer: {args.buffer}")
    print(f"Games stored: {len(buf)}  |  Total games ever seen: {data['total_games']}")

    lengths = [len(g) for g in buf]
    outcomes = [g.game_outcome for g in buf]
    print(f"Length:  min {min(lengths)}  max {max(lengths)}  avg {sum(lengths)/len(lengths):.1f}")
    print(f"Outcomes (p1 perspective): {dict(Counter(outcomes))}")

    total_moves = 0
    total_caps = 0
    total_checks = 0
    illegal_games = 0
    per_game_caps = []
    for g in buf:
        _, caps, checks, ok = replay_game(g)
        total_moves += len(g.actions)
        total_caps += caps
        total_checks += checks
        per_game_caps.append(caps)
        if not ok:
            illegal_games += 1

    print(f"Moves:     {total_moves}")
    print(f"Captures:  {total_caps}  ({100 * total_caps / max(total_moves, 1):.2f}% of moves)")
    print(f"Checks:    {total_checks}  ({100 * total_checks / max(total_moves, 1):.2f}% of moves)")
    print(f"Per-game captures: avg {sum(per_game_caps)/len(per_game_caps):.1f}  "
          f"min {min(per_game_caps)}  max {max(per_game_caps)}  "
          f"zero-capture games: {sum(1 for c in per_game_caps if c == 0)}/{len(per_game_caps)}")
    if illegal_games:
        print(f"WARNING: {illegal_games} games had illegal actions when replayed")

    if args.summary_only:
        return

    print()
    for i, g in enumerate(buf[:args.num_games]):
        san, caps, checks, _ = replay_game(g)
        print(f"--- Game {i}  outcome={g.game_outcome}  plies={len(g)}  captures={caps}  checks={checks}")
        print(format_pgn_line(san))
        print()


if __name__ == "__main__":
    main()
