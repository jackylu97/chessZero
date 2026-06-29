import sys
import numpy as np
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.games.chess import ChessGame, GameState, _move_to_action
from src.games.chess_gpu import GpuChessGame


def cpu_obs(board):
    g = ChessGame()
    st = GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1)
    return g.to_tensor(st).numpy()


gpu = GpuChessGame()
board = chess.Board()
gstate = gpu.from_python_chess([chess.Board()], device="cpu")

# play b2b4 (ply1), h7h5(ply2) deterministically to reach the mismatch fen at ply 2
moves = ["b2b4", "h7h5"]
for uci in moves:
    mv = chess.Move.from_uci(uci)
    action = _move_to_action(mv, board)
    board.push(mv)
    gstate, _, _ = gpu.step_batch(gstate, torch.tensor([action], dtype=torch.int64))

print("board fen:", board.fen(), " turn:", "white" if board.turn else "black")
print("gpu side:", int(gstate.side[0]), " (0=white,1=black)")
print("gpu fullmove:", int(gstate.fullmove[0]), " halfmove:", int(gstate.halfmove[0]))

c = cpu_obs(board.copy())
g = gpu.to_tensor_batch(gstate)[0].numpy()

for p in [0, 6, 16, 21]:
    print(f"\n=== plane {p} ===")
    print("CPU:\n", c[p].astype(int) if p != 21 else c[p])
    print("GPU:\n", g[p].astype(int) if p != 21 else g[p])
