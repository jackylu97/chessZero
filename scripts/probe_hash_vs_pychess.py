"""Compare GPU _zobrist_hash equivalence classes vs python-chess transposition
key (what is_repetition uses). Pin whether the off-by-one is a phase issue or a
hash-collision/definition mismatch.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.games.chess import _move_to_action
from src.games.chess_gpu import GpuChessGame, _zobrist_hash

device = "cuda"
gpu = GpuChessGame(); gpu.max_plies = 60
shuffle = ["e2e4","e7e5","g1f3","b8c6","f3g1","c6b8","g1f3","b8c6","f3g1","c6b8"]
state = gpu.reset_batch(1, device=device)
next_legal_mask = gpu.legal_mask(state)
board = chess.Board()

rows = []
for ply in range(len(shuffle)):
    gpu_h = int(_zobrist_hash(state)[0].item())
    py_key = board._transposition_key()
    rows.append((ply, gpu_h, py_key))
    mv = chess.Move.from_uci(shuffle[ply]); a = _move_to_action(mv, board.turn)
    board.push(mv)
    at = torch.tensor([a], dtype=torch.int64, device=device)
    state, _, _, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        break

print("ply | gpu_hash equiv-class | pychess transposition-key equiv-class")
# Build equivalence-class labels
from collections import OrderedDict
def labels(vals):
    seen = OrderedDict(); out = []
    for v in vals:
        if v not in seen:
            seen[v] = len(seen)
        out.append(seen[v])
    return out
gpu_lab = labels([r[1] for r in rows])
py_lab = labels([r[2] for r in rows])
for i, (ply, gh, pk) in enumerate(rows):
    flag = "" if gpu_lab[i] == py_lab[i] else "  <<< CLASS MISMATCH"
    print(f"{ply:3d} |        {gpu_lab[i]}        |          {py_lab[i]}{flag}")

print("\nGPU equivalence classes:", gpu_lab)
print("pychess equivalence classes:", py_lab)
