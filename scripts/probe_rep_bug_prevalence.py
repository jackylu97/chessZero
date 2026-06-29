"""Quantify prevalence of the ep-phantom repetition bug in the REAL buffer.

For each ply of real games, compute:
  - GPU rep planes (replay through GpuChessGame, reading to_tensor_batch)
  - CPU rep planes (python-chess board.is_repetition via ChessGame.to_tensor)
and count plies where they DISAGREE on rep>=2 or rep>=3. Also count how often
the disagreement happens at the actual repetition positions.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.games.chess_gpu import GpuChessGame

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
device = "cuda"
gpu = GpuChessGame(); gpu.max_plies = 1100
cpu = ChessGame()

N_GAMES = 60
disagree_r2 = 0
disagree_r3 = 0
total_plies = 0
games = 0
games_with_disagreement = 0
# also: does the bug change the threefold TERMINATION ply?

with open(BUF, "rb") as f:
    header = pickle.load(f)
    nrec = header["n_records"]
    read = 0
    for r in range(nrec):
        rec, prio = pickle.load(f)
        if not bool(rec.get("draw_by_repetition", False)):
            continue
        actions = [int(a) for a in rec["actions"]]
        # Replay on GPU (batch of 1) and on CPU board in lockstep.
        state = gpu.reset_batch(1, device=device)
        next_legal = gpu.legal_mask(state)
        board = chess.Board()
        gdis = False
        for a in actions:
            so = gpu.to_tensor_batch(state)
            g_r2 = float(so[0, 19, 0, 0]); g_r3 = float(so[0, 20, 0, 0])
            c_r2 = 1.0 if board.is_repetition(2) else 0.0
            c_r3 = 1.0 if board.is_repetition(3) else 0.0
            total_plies += 1
            if g_r2 != c_r2:
                disagree_r2 += 1; gdis = True
            if g_r3 != c_r3:
                disagree_r3 += 1; gdis = True
            mv = _action_to_move(a, board)
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
            at = torch.tensor([_move_to_action(mv, not board.turn)], dtype=torch.int64, device=device)
            # the action we pushed corresponds to the pre-push side; re-derive via stored a
            at = torch.tensor([a], dtype=torch.int64, device=device)
            state, _, _, next_legal = gpu.step_batch_with_legal(state, at)
            if bool(state.done):
                break
        games += 1
        if gdis:
            games_with_disagreement += 1
        read += 1
        if read >= N_GAMES:
            break

print(f"games checked (threefold-drawn): {games}")
print(f"total plies: {total_plies}")
print(f"plies where GPU(self-play) rep>=2 != CPU(train) rep>=2: {disagree_r2} "
      f"({100*disagree_r2/max(total_plies,1):.1f}%)")
print(f"plies where GPU(self-play) rep>=3 != CPU(train) rep>=3: {disagree_r3} "
      f"({100*disagree_r3/max(total_plies,1):.1f}%)")
print(f"games with >=1 rep-plane disagreement: {games_with_disagreement}/{games} "
      f"({100*games_with_disagreement/max(games,1):.1f}%)")
