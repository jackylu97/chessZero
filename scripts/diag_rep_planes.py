"""Diagnostic: compare repetition / no-progress obs planes between the CPU
ChessGame.to_tensor and the GPU GpuChessGame.to_tensor_batch, by STEPPING both
engines through an identical repetition sequence (the way self-play does).

Dumps planes 19 (rep>=2), 20 (rep>=3), 21 (no-progress) at each ply.
"""
import sys
import numpy as np
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.games.chess import ChessGame, GameState, _move_to_action
from src.games.chess_gpu import GpuChessGame


def cpu_planes(board):
    g = ChessGame()
    st = GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1)
    t = g.to_tensor(st).numpy()
    return t[19, 0, 0], t[20, 0, 0], t[21, 0, 0]


def main():
    cpu_game = ChessGame()
    gpu_game = GpuChessGame()

    # A shuffle that repeats the start-ish position 3 times:
    # Ng1-f3 Ng8-f6 Nf3-g1 Nf6-g8  (repeats) ... do it three cycles.
    moves = [
        "g1f3", "g8f6", "f3g1", "f6g8",   # cycle -> position after this == start (2nd occurrence of start)
        "g1f3", "g8f6", "f3g1", "f6g8",   # -> start 3rd occurrence
        "g1f3", "g8f6", "f3g1", "f6g8",   # -> start 4th occurrence
    ]

    # CPU board with full move stack.
    cpu_board = chess.Board()
    # GPU state stepped from start.
    gpu_state = gpu_game.from_python_chess([chess.Board()], device="cpu")

    print(f"{'ply':>3} {'move':>6} | {'fen-ish':<20} | CPU rep2 rep3 np  |  GPU rep2 rep3 np  | py-chess is_rep(2/3)")
    # initial
    c2, c3, cnp = cpu_planes(cpu_board.copy())
    gt = gpu_game.to_tensor_batch(gpu_state)[0].numpy()
    g2, g3, gnp = gt[19, 0, 0], gt[20, 0, 0], gt[21, 0, 0]
    print(f"{0:>3} {'(init)':>6} | {'startpos':<20} | {c2:.1f}  {c3:.1f}  {cnp:.3f} |   {g2:.1f}  {g3:.1f}  {gnp:.3f} | {cpu_board.is_repetition(2)}/{cpu_board.is_repetition(3)}")

    mismatches = []
    for i, uci in enumerate(moves, start=1):
        mv = chess.Move.from_uci(uci)
        action = _move_to_action(mv, cpu_board.turn)
        # step CPU
        cpu_board.push(mv)
        # step GPU
        act_t = torch.tensor([action], dtype=torch.int64)
        gpu_state, _, _ = gpu_game.step_batch(gpu_state, act_t)

        c2, c3, cnp = cpu_planes(cpu_board.copy())
        gt = gpu_game.to_tensor_batch(gpu_state)[0].numpy()
        g2, g3, gnp = gt[19, 0, 0], gt[20, 0, 0], gt[21, 0, 0]
        ir2, ir3 = cpu_board.is_repetition(2), cpu_board.is_repetition(3)
        tag = ""
        if abs(c2 - g2) > 1e-6 or abs(c3 - g3) > 1e-6 or abs(cnp - gnp) > 1e-6:
            tag = "  <<< MISMATCH"
            mismatches.append(i)
        print(f"{i:>3} {uci:>6} | {'':<20} | {c2:.1f}  {c3:.1f}  {cnp:.3f} |   {g2:.1f}  {g3:.1f}  {gnp:.3f} | {ir2}/{ir3}{tag}")

    print()
    print(f"mismatches at plies: {mismatches}")

    # --- Also: build GPU state DIRECTLY from a board that already has a repetition
    #     in its move stack (the reanalyze / from_python_chess path).
    print("\n=== from_python_chess on a board with repetition already in stack ===")
    gpu_from_board = gpu_game.from_python_chess([cpu_board.copy()], device="cpu")
    gt = gpu_game.to_tensor_batch(gpu_from_board)[0].numpy()
    c2, c3, cnp = cpu_planes(cpu_board.copy())
    print(f"CPU (full stack):       rep2={c2:.1f} rep3={c3:.1f} np={cnp:.3f}  is_rep(2/3)={cpu_board.is_repetition(2)}/{cpu_board.is_repetition(3)}")
    print(f"GPU (from_python_chess): rep2={gt[19,0,0]:.1f} rep3={gt[20,0,0]:.1f} np={gt[21,0,0]:.3f}")


if __name__ == "__main__":
    main()
