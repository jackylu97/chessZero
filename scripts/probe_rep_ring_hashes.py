"""Dump the actual ring contents + current hash at ply 6 to see why the
post-e4-e5 hash is only counted once when it occurred twice.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame, _zobrist_hash, REP_HISTORY_K

device = "cuda"
gpu = GpuChessGame(); gpu.max_plies = 60
shuffle = ["e2e4","e7e5","g1f3","b8c6","f3g1","c6b8","g1f3","b8c6","f3g1","c6b8"]
state = gpu.reset_batch(1, device=device)
next_legal_mask = gpu.legal_mask(state)
board = chess.Board()

hashes_seen = []  # (ply, current_hash)
for ply in range(len(shuffle)):
    cur_hash = int(_zobrist_hash(state)[0].item())
    rc = int(state.rep_count[0].item())
    ring = state.rep_hashes[0, :max(rc, 1) + 2].tolist()
    hashes_seen.append((ply, cur_hash))
    print(f"ply {ply} ({shuffle[ply]}): cur_hash={cur_hash}  rep_count={rc}")
    print(f"     ring[0:{rc+2}] = {ring}")
    mv = chess.Move.from_uci(shuffle[ply]); a = _move_to_action(mv, board.turn)
    board.push(mv)
    at = torch.tensor([a], dtype=torch.int64, device=device)
    state, _, _, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        break

print("\nCurrent-hash sequence by ply:")
for ply, h in hashes_seen:
    print(f"  ply {ply}: {h}")
# Which plies share the same hash (true repetitions)?
from collections import Counter
cnt = Counter(h for _, h in hashes_seen)
print("\nHashes occurring >1 time across the traced plies:")
for h, n in cnt.items():
    if n > 1:
        plies = [p for p, hh in hashes_seen if hh == h]
        print(f"  hash {h} at plies {plies} (n={n})")
