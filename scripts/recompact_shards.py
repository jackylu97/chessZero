#!/usr/bin/env python3
"""Recompact legacy full-observation Stockfish shards into the compact v2 format.

The asymmetric pool generated 2026-06-12/13 was written as legacy
``list[GameHistory]`` pickles carrying full observation tensors (~3.7 MB/game)
instead of the compact v2 shard format ``generate_stockfish_games.py`` was
supposed to emit (~3.4 KB/game). This reclaims ~1000x disk with ZERO game loss:
observations + legal_actions_list are dropped and reconstructed at load time by
replaying actions through ChessGame (exactly what ``_iter_shard_games`` does for
v2 shards).

Per shard: read legacy -> write compact temp in the same dir -> verify
(count + full round-trip of first & last game) -> atomic os.replace -> the
legacy bytes are freed. Idempotent: shards already in v2 format are skipped, so
re-running after an interruption is safe. Processes one shard at a time to bound
peak RAM (~2 GB/legacy-shard).
"""

import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory, _iter_shard_games


def shard_is_compact(path: str) -> bool:
    with open(path, "rb") as f:
        first = pickle.load(f)
    return isinstance(first, dict) and first.get("version") == 2


def recompact_one(path: str, game) -> tuple[str, int, int]:
    """Returns (status, n_games, bytes_freed)."""
    orig_size = os.path.getsize(path)
    if orig_size == 0:
        return ("empty", 0, 0)

    with open(path, "rb") as f:
        first = pickle.load(f)
        if isinstance(first, dict) and first.get("version") == 2:
            return ("skip", int(first.get("n_records", 0)), 0)
        if not isinstance(first, list):
            return ("bad", 0, 0)
        games = first  # list[GameHistory]

    n = len(games)
    compact = [g.to_compact_dict() for g in games]

    tmp = path + ".compact.tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"version": 2, "n_records": n}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for d in compact:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Verify before destroying the original ---
    # Cheap full count: re-read raw records (no replay) and confirm N + schema.
    with open(tmp, "rb") as f:
        header = pickle.load(f)
        assert header.get("version") == 2 and header["n_records"] == n, "header mismatch"
        for _ in range(n):
            d = pickle.load(f)
            assert isinstance(d, dict) and d.get("format_version") == 3, "record schema"
    # Full round-trip (replay through ChessGame) for first & last game only.
    for orig, dec in (
        (games[0], GameHistory.from_compact_dict(compact[0], game)),
        (games[-1], GameHistory.from_compact_dict(compact[-1], game)),
    ):
        assert [int(a) for a in orig.actions] == [int(a) for a in dec.actions], "actions"
        assert abs(float(orig.game_outcome) - float(dec.game_outcome)) < 1e-6, "outcome"
        assert len(dec.observations) == len(dec.actions) + 1, "obs count"
        if orig.external_values:
            assert len(dec.external_values) == len(orig.external_values), "ext_values len"

    new_size = os.path.getsize(tmp)
    os.replace(tmp, path)  # same-dir rename: atomic, frees the legacy bytes
    return ("ok", n, orig_size - new_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/stockfish_injection",
                    help="Directory tree of .pkl shards to recompact in place.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be done without writing/deleting.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "*.pkl"), recursive=True))
    print(f"Found {len(paths)} shard(s) under {args.root}", flush=True)

    game = ChessGame()
    tot_games = tot_freed = n_ok = n_skip = n_bad = n_empty = 0
    for i, p in enumerate(paths, 1):
        if args.dry_run:
            tag = "compact" if shard_is_compact(p) else "legacy"
            print(f"[{i}/{len(paths)}] {tag:7s} {os.path.getsize(p)/1e6:8.1f} MB  {p}", flush=True)
            continue
        try:
            status, n, freed = recompact_one(p, game)
        except Exception as e:  # never destroy a shard we couldn't verify
            print(f"[{i}/{len(paths)}] ERROR {p}: {type(e).__name__}: {e}", flush=True)
            tmp = p + ".compact.tmp"
            if os.path.exists(tmp):
                os.remove(tmp)
            n_bad += 1
            continue
        if status == "ok":
            n_ok += 1; tot_games += n; tot_freed += freed
            print(f"[{i}/{len(paths)}] ok    {n:4d} games  freed {freed/1e6:8.1f} MB  "
                  f"(cum {tot_freed/1e9:6.2f} GB)  {p}", flush=True)
        elif status == "skip":
            n_skip += 1; print(f"[{i}/{len(paths)}] skip (already compact)  {p}", flush=True)
        elif status == "empty":
            n_empty += 1; print(f"[{i}/{len(paths)}] empty (0 bytes)  {p}", flush=True)
        else:
            n_bad += 1; print(f"[{i}/{len(paths)}] bad ({status})  {p}", flush=True)

    print(f"\nDONE: {n_ok} recompacted ({tot_games} games), {n_skip} already-compact, "
          f"{n_empty} empty, {n_bad} errored. Reclaimed {tot_freed/1e9:.2f} GB.", flush=True)


if __name__ == "__main__":
    main()
