"""Confirm the ep-square root cause: GPU hashes ep whenever ep>=0, but
python-chess only includes ep in the transposition key when an en-passant
capture is actually LEGAL. So a position with a 'phantom' ep square (no legal
ep capture) is treated as DISTINCT by the GPU but IDENTICAL by python-chess,
breaking repetition detection.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.games.chess import _move_to_action
from src.games.chess_gpu import GpuChessGame

device = "cuda"
gpu = GpuChessGame(); gpu.max_plies = 60
shuffle = ["e2e4","e7e5","g1f3","b8c6","f3g1","c6b8","g1f3","b8c6","f3g1","c6b8"]
state = gpu.reset_batch(1, device=device)
next_legal_mask = gpu.legal_mask(state)
board = chess.Board()

print("ply | GPU state.ep | pychess ep_square | pychess has_legal_en_passant | "
      "in transposition key?")
for ply in range(len(shuffle)):
    gpu_ep = int(state.ep[0].item())
    py_ep = board.ep_square
    legal_ep = board.has_legal_en_passant()
    # transposition key includes ep only if has_legal_en_passant
    key_ep = py_ep if legal_ep else None
    print(f"{ply:3d} |     {gpu_ep:3d}      |       {str(py_ep):5s}       |"
          f"        {str(legal_ep):5s}            | {key_ep}")
    mv = chess.Move.from_uci(shuffle[ply]); a = _move_to_action(mv, board.turn)
    board.push(mv)
    at = torch.tensor([a], dtype=torch.int64, device=device)
    state, _, _, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        break

print("\nAt ply 2: GPU ep set (e6=44), but python-chess has_legal_en_passant=False")
print(" -> python-chess key drops ep, GPU keeps it -> GPU sees ply2 != ply6.")
