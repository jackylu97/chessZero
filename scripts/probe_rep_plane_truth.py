"""Isolate the rep>=2 plane (19) divergence: GPU Zobrist-ring vs CPU
is_repetition, vs the GROUND TRUTH (python-chess board.is_repetition counting
actual occurrences). Which path is wrong?
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame

device = "cuda"
gpu = GpuChessGame(); gpu.max_plies = 60
cpu = ChessGame()

shuffle = ["e2e4","e7e5","g1f3","b8c6","f3g1","c6b8","g1f3","b8c6","f3g1","c6b8"]

state = gpu.reset_batch(1, device=device)
next_legal_mask = gpu.legal_mask(state)
board = chess.Board()

print("ply move  | GPU rep>=2  GPU rep>=3 | CPU(to_tensor) rep>=2 rep>=3 | "
      "truth: #occurrences of resulting pos")
for ply in range(len(shuffle)):
    # GPU obs of CURRENT (pre-move) state
    so = gpu.to_tensor_batch(state)
    gpu_r2 = float(so[0, 19, 0, 0]); gpu_r3 = float(so[0, 20, 0, 0])

    # CPU to_tensor of the SAME pre-move board
    st = cpu.reset(); st.board = board.copy()
    co = cpu.to_tensor(st)
    cpu_r2 = float(co[19, 0, 0]); cpu_r3 = float(co[20, 0, 0])

    # ground truth occurrence count of CURRENT position in the move history
    # python-chess is_repetition(n): current pos appeared >= n times.
    truth2 = board.is_repetition(2)
    truth3 = board.is_repetition(3)

    print(f"{ply:3d} {shuffle[ply]:5s} |   {gpu_r2:.0f}        {gpu_r3:.0f}     |"
          f"       {cpu_r2:.0f}        {cpu_r3:.0f}    | "
          f"is_rep2={truth2} is_rep3={truth3}")

    mv = chess.Move.from_uci(shuffle[ply]); a = _move_to_action(mv, board.turn)
    board.push(mv)
    at = torch.tensor([a], dtype=torch.int64, device=device)
    state, _, _, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        print(f"  (GPU game ended at ply {ply}, threefold={bool(state.terminal_threefold[0])})")
        break
