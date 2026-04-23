"""Sanity-check the MultiPV softmax sampler in generate_stockfish_games.py.

Generates N games and reports:
  1. Per-game summary: length, outcome, eval-gap distribution (best_eval - chosen_eval).
  2. Any plies with gap > threshold (default 0.3 in [-1,+1] value space = ~120cp).
  3. Cross-game divergence: first ply at which each pair of games differs.

Run:
  .venv/bin/python scripts/test_stockfish_diversity.py \\
      --num-games 3 --depth 8 --multipv 3 --move-temperature 0.15
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
import numpy as np

from src.games.chess import ChessGame

# Reuse the production helpers — we're testing the same code path the training
# generation will use, not a separate re-implementation.
from scripts.generate_stockfish_games import generate_one_game


def _game_move_sequence(history) -> list[int]:
    """Return the action list (ints) for comparing games move-by-move."""
    return list(history.actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--num-games", type=int, default=3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--move-temperature", type=float, default=0.15)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--gap-threshold", type=float, default=0.3,
                    help="Flag any ply where (best_eval - chosen_eval) exceeds this.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stockfish-threads", type=int, default=1)
    ap.add_argument("--stockfish-hash", type=int, default=64)
    args = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        engine.configure(
            {"Threads": args.stockfish_threads, "Hash": args.stockfish_hash}
        )
    except chess.engine.EngineError as e:
        print(f"Warning: {e}")

    game = ChessGame()
    rng = random.Random(args.seed)

    histories = []
    per_game_gaps = []

    print(f"Generating {args.num_games} games @ depth={args.depth}, "
          f"multipv={args.multipv}, τ={args.move_temperature}\n")

    try:
        for i in range(args.num_games):
            gaps: list[float] = []
            h = generate_one_game(
                engine, game, args.depth, args.max_plies,
                args.multipv, args.move_temperature, rng,
                gap_sink=gaps,
            )
            if h is None:
                print(f"Game {i}: hit ply cap, discarded.")
                continue
            histories.append(h)
            per_game_gaps.append(gaps)

            gaps_arr = np.array(gaps)
            outcome = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}.get(h.game_outcome, "?")
            flagged = int((gaps_arr > args.gap_threshold).sum())

            print(f"Game {i}: length={len(h.actions)} plies, result={outcome}")
            print(f"  eval-gap (best - chosen) distribution:")
            print(f"    mean={gaps_arr.mean():+.4f}  median={np.median(gaps_arr):+.4f}  "
                  f"std={gaps_arr.std():.4f}  max={gaps_arr.max():+.4f}")
            print(f"    percentiles: p50={np.percentile(gaps_arr, 50):+.4f}  "
                  f"p90={np.percentile(gaps_arr, 90):+.4f}  "
                  f"p99={np.percentile(gaps_arr, 99):+.4f}")
            print(f"  plies with gap > {args.gap_threshold}: {flagged} / {len(gaps)}")

            if flagged > 0:
                print(f"  flagged plies (ply : gap : best_eval : chosen_eval):")
                # Reconstruct context: we have root_values (= chosen_eval) in history.
                # best_eval = chosen + gap.
                for ply_idx, g in enumerate(gaps):
                    if g > args.gap_threshold:
                        chosen = h.root_values[ply_idx]
                        print(f"    ply {ply_idx:3d}: gap={g:+.4f}  "
                              f"best={chosen + g:+.4f}  chosen={chosen:+.4f}")
            print()
    finally:
        engine.quit()

    # Cross-game divergence.
    if len(histories) >= 2:
        print("Cross-game divergence:")
        for i in range(len(histories)):
            for j in range(i + 1, len(histories)):
                seq_i = _game_move_sequence(histories[i])
                seq_j = _game_move_sequence(histories[j])
                first_diff = None
                for k in range(min(len(seq_i), len(seq_j))):
                    if seq_i[k] != seq_j[k]:
                        first_diff = k
                        break
                if first_diff is None and len(seq_i) == len(seq_j):
                    print(f"  games {i}↔{j}: IDENTICAL over {len(seq_i)} plies (!!)")
                elif first_diff is None:
                    print(f"  games {i}↔{j}: one is prefix of other (lengths "
                          f"{len(seq_i)}, {len(seq_j)})")
                else:
                    print(f"  games {i}↔{j}: first divergence at ply {first_diff}")

    # Aggregate gap stats across all games.
    if per_game_gaps:
        all_gaps = np.concatenate([np.array(g) for g in per_game_gaps])
        total_flagged = int((all_gaps > args.gap_threshold).sum())
        print(f"\nAggregate over {len(per_game_gaps)} games, {len(all_gaps)} plies:")
        print(f"  mean gap: {all_gaps.mean():+.4f}, max: {all_gaps.max():+.4f}")
        print(f"  plies with gap > {args.gap_threshold}: "
              f"{total_flagged} ({100 * total_flagged / len(all_gaps):.1f}%)")


if __name__ == "__main__":
    main()
