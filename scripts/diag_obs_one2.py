import sys
import numpy as np
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.games.chess import ChessGame, GameState, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.games.chess_gpu import _zobrist_hash

gpu = GpuChessGame()

# Test 1: just b2b4 then h7h5
for seq in (["b2b4", "h7h5"], ["e2e4", "e7e5"], ["h2h4", "h7h5"]):
    board = chess.Board()
    gstate = gpu.from_python_chess([chess.Board()], device="cpu")
    for uci in seq:
        mv = chess.Move.from_uci(uci)
        action = _move_to_action(mv, board)
        board.push(mv)
        gstate, _, done = gpu.step_batch(gstate, torch.tensor([action], dtype=torch.int64))
    # compare ep
    cpu_ep = board.ep_square
    gpu_ep = int(gstate.ep[0])
    # black pawn bitboard (plane index 6)
    gpu_bp = int(gstate.pieces[0, 6])  # black pawns (color_offset 6 + plane 0 = pawns)
    real_bp = int(board.pieces(chess.PAWN, chess.BLACK))
    print(f"seq {seq}: fen={board.fen()}")
    print(f"   CPU ep_square={cpu_ep}  GPU ep={gpu_ep}   (match={cpu_ep==(gpu_ep if gpu_ep>=0 else None)})")
    print(f"   black-pawn bb: GPU={gpu_bp:016x}  real={real_bp:016x}  match={gpu_bp==real_bp}")
    # white pawn bb
    gpu_wp = int(gstate.pieces[0, 0])
    real_wp = int(board.pieces(chess.PAWN, chess.WHITE))
    print(f"   white-pawn bb: GPU={gpu_wp:016x}  real={real_wp:016x}  match={gpu_wp==real_wp}")
    print()
