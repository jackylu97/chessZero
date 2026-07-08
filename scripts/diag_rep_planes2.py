"""Diag 2: confirm WHEN the GPU game is marked done in the shuffle, and check
the no-progress divergence is purely the post-done freeze (benign) vs a live bug.
Also test the reanalyze path: does reanalyze rebuild GPU state per-position
(losing rep history) or step through?
"""
import sys
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame


def main():
    gpu_game = GpuChessGame()
    gpu_state = gpu_game.from_python_chess([chess.Board()], device="cpu")
    cpu_board = chess.Board()

    moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 3
    print(f"{'ply':>3} {'move':>6} | GPU done | GPU halfmove | reward | CPU is_rep(3) | CPU outcome")
    for i, uci in enumerate(moves, start=1):
        mv = chess.Move.from_uci(uci)
        action = _move_to_action(mv, cpu_board.turn)
        cpu_board.push(mv)
        act_t = torch.tensor([action], dtype=torch.int64)
        gpu_state, reward, done = gpu_game.step_batch(gpu_state, act_t)
        oc = cpu_board.outcome(claim_draw=False)
        print(f"{i:>3} {uci:>6} | {bool(done[0])!s:>8} | {int(gpu_state.halfmove[0]):>12} | "
              f"{float(reward[0]):>6.1f} | {cpu_board.is_repetition(3)!s:>13} | {oc}")

    print()
    print("Note: python-chess board.outcome(claim_draw=False) does NOT auto-claim "
          "threefold (only fivefold/75-move are automatic). GPU uses strict-3fold "
          "auto-terminal. This is a deliberate engine difference.")


if __name__ == "__main__":
    main()
