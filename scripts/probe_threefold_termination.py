"""Does the GPU threefold TERMINATION ply match python-chess's? The buggy
ep-in-hash could make the GPU detect threefold later (or differently) than the
true rules, inflating shuffle length. Replay real games and compare the ply at
which each path first sees a true threefold.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.games.chess import _action_to_move
from src.games.chess_gpu import GpuChessGame

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
device = "cuda"
gpu = GpuChessGame(); gpu.max_plies = 1100

N = 60
gpu_term, py_term = [], []
mismatches = 0
with open(BUF, "rb") as f:
    header = pickle.load(f)
    read = 0
    for r in range(header["n_records"]):
        rec, prio = pickle.load(f)
        if not bool(rec.get("draw_by_repetition", False)):
            continue
        actions = [int(a) for a in rec["actions"]]
        state = gpu.reset_batch(1, device=device)
        board = chess.Board()
        gpu_t = None; py_t = None
        for i, a in enumerate(actions):
            mv = _action_to_move(a, board)
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
            if py_t is None and board.is_repetition(3):
                py_t = i + 1  # ply count after this move
            at = torch.tensor([a], dtype=torch.int64, device=device)
            state, _, done, _ = gpu.step_batch_with_legal(state, at)
            if gpu_t is None and bool(state.terminal_threefold[0]):
                gpu_t = i + 1
            if bool(state.done):
                break
        gpu_term.append(gpu_t); py_term.append(py_t)
        if gpu_t != py_t:
            mismatches += 1
        read += 1
        if read >= N:
            break

# Stats: where both detected, compare.
both = [(g, p) for g, p in zip(gpu_term, py_term) if g is not None and p is not None]
diffs = [g - p for g, p in both]
print(f"games: {len(gpu_term)}")
print(f"GPU detected threefold: {sum(1 for g in gpu_term if g is not None)}")
print(f"python-chess detected threefold: {sum(1 for p in py_term if p is not None)}")
print(f"termination-ply mismatches (gpu_t != py_t): {mismatches}")
if diffs:
    arr = np.array(diffs)
    print(f"gpu_t - py_t: mean={arr.mean():.2f} min={arr.min()} max={arr.max()} "
          f"(positive = GPU terminates LATER than the true rule)")
    print(f"  games where GPU terminated >=2 plies late: {int((arr>=2).sum())}")
