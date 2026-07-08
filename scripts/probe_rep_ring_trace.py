"""Trace the GPU rep ring: rep_count, matches_count computed by to_tensor_batch,
vs ground truth occurrence count, ply by ply. Pin the off-by-one root cause.
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

def gt_count(b):
    """How many times the current position has occurred (>= ... )."""
    # python-chess: is_repetition(n) True iff appeared >= n times.
    for n in range(5, 0, -1):
        if b.is_repetition(n):
            return n
    return 1

print("Tracing CURRENT (pre-move) state each ply:")
print("ply move  | rep_count | matches_count(to_tensor) | rep>=2plane | GT occurrences")
for ply in range(len(shuffle)):
    cur_hash = _zobrist_hash(state)
    rc = int(state.rep_count[0].item())
    rep_pos = torch.arange(REP_HISTORY_K, device=device).unsqueeze(0)
    valid = rep_pos < state.rep_count.to(torch.int64).unsqueeze(1)
    matches = (state.rep_hashes == cur_hash.unsqueeze(1)) & valid
    mc = int(matches.sum(dim=1)[0].item())

    so = gpu.to_tensor_batch(state)
    r2 = float(so[0, 19, 0, 0])

    gt = gt_count(board)
    flag = "" if ((mc >= 2) == (gt >= 2)) else "  <<< OFF BY ONE"
    print(f"{ply:3d} {shuffle[ply]:5s} |    {rc:2d}     |          {mc:2d}            |     "
          f"{r2:.0f}      |   GT={gt}{flag}")

    mv = chess.Move.from_uci(shuffle[ply]); a = _move_to_action(mv, board.turn)
    board.push(mv)
    at = torch.tensor([a], dtype=torch.int64, device=device)
    state, _, _, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        break
