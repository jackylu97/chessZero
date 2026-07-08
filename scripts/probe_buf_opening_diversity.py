"""Probe: buffer opening diversity. With random_opening_plies=0, are openings
degenerate (same first moves every game)? Low diversity -> buffer can't teach
the policy to distinguish positions.

Reports, over self-play games in a .buf:
  - distribution of first move, first 2 plies, first 4 plies (unique counts)
  - entropy of the first-move distribution
  - mean game length, draw types
  - per-ply position uniqueness (how many distinct positions at ply k)
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory


def load_games(buf_path, game, max_games=2000):
    import pickle
    games = []
    with open(buf_path, "rb") as f:
        first = pickle.load(f)
        version = first["version"]; n = first["n_records"]
        for _ in range(n):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if len(games) >= max_games:
                break
    return games


def entropy(counter):
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    ps = np.array([c / tot for c in counter.values()])
    return float(-(ps * np.log(ps + 1e-12)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/chess/2026_06_19_cold2_pc")
    ap.add_argument("--bufs", nargs="*", default=None)
    ap.add_argument("--max-games", type=int, default=2000)
    args = ap.parse_args()

    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])

    bufs = args.bufs or [
        os.path.join(args.ckpt_dir, f"checkpoint_{s}.buf") for s in (6000, 30000)
    ]
    for buf_path in bufs:
        if not os.path.exists(buf_path):
            print(f"(missing) {buf_path}"); continue
        games = load_games(buf_path, game, max_games=args.max_games)
        sp = [g for g in games if not g.external_values and len(g.actions) >= 4]
        print(f"\n{'='*70}\n{os.path.basename(buf_path)}: {len(games)} games, {len(sp)} self-play\n{'='*70}")
        if not sp:
            continue
        m1 = Counter(g.actions[0] for g in sp)
        m2 = Counter(tuple(g.actions[:2]) for g in sp)
        m4 = Counter(tuple(g.actions[:4]) for g in sp)
        m8 = Counter(tuple(g.actions[:8]) for g in sp)
        lens = [len(g.actions) for g in sp]
        rep = sum(1 for g in sp if g.draw_by_repetition)
        nop = sum(1 for g in sp if g.draw_by_no_progress)
        print(f"  game length: mean={np.mean(lens):.1f} med={np.median(lens):.0f} "
              f"min={min(lens)} max={max(lens)}")
        print(f"  draw_by_repetition={rep}/{len(sp)} draw_by_no_progress={nop}/{len(sp)}")
        print(f"  unique 1st move:   {len(m1):4d}  entropy={entropy(m1):.3f} "
              f"(max possible ~{np.log(20):.2f})  top: {m1.most_common(1)[0][1]}/{len(sp)}")
        print(f"  unique 1st 2 ply:  {len(m2):4d}  entropy={entropy(m2):.3f}")
        print(f"  unique 1st 4 ply:  {len(m4):4d}  entropy={entropy(m4):.3f}")
        print(f"  unique 1st 8 ply:  {len(m8):4d}  entropy={entropy(m8):.3f}")
        # top-5 most common openings (first 4 plies)
        print("  top-5 4-ply openings (action tuples): ")
        for tup, c in m4.most_common(5):
            print(f"      {c:4d}/{len(sp)}  {tup}")
        # position uniqueness by ply: distinct action-prefixes seen at ply k
        for k in (2, 6, 10, 16):
            prefixes = Counter(tuple(g.actions[:k]) for g in sp if len(g.actions) >= k)
            ngames = sum(1 for g in sp if len(g.actions) >= k)
            print(f"  ply {k:2d}: {len(prefixes):4d} distinct prefixes among {ngames} games "
                  f"({100*len(prefixes)/max(ngames,1):.0f}% unique)")


if __name__ == "__main__":
    main()
