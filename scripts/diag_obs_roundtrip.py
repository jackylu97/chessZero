"""Full 22-plane obs round-trip: step CPU ChessGame and GPU GpuChessGame through
identical random games and assert all planes match at every ply. Catches any
numpy<->tensor obs corruption (would corrupt training)."""
import sys
import numpy as np
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.games.chess import ChessGame, GameState, _move_to_action, _action_to_move
from src.games.chess_gpu import GpuChessGame


def cpu_obs(board):
    g = ChessGame()
    st = GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1)
    return g.to_tensor(st).numpy()


def main():
    rng = np.random.default_rng(7)
    gpu = GpuChessGame()
    n_games = 30
    max_mismatch_report = 10
    total_plies = 0
    per_plane_maxdiff = np.zeros(22)
    mism = 0
    for gi in range(n_games):
        board = chess.Board()
        gstate = gpu.from_python_chess([chess.Board()], device="cpu")
        for ply in range(80):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            # compare BEFORE moving
            c = cpu_obs(board.copy())
            g = gpu.to_tensor_batch(gstate)[0].numpy()
            d = np.abs(c - g)
            per_plane_maxdiff = np.maximum(per_plane_maxdiff, d.reshape(22, -1).max(1))
            total_plies += 1
            if d.max() > 1e-5:
                mism += 1
                if mism <= max_mismatch_report:
                    bad = np.where(d.reshape(22, -1).max(1) > 1e-5)[0]
                    print(f"game {gi} ply {ply}: MISMATCH planes {bad.tolist()} "
                          f"maxdiff={d.max():.3f}  fen={board.fen()}")
            mv = legal[int(rng.integers(len(legal)))]
            action = _move_to_action(mv, board.turn)
            board.push(mv)
            gstate, _, done = gpu.step_batch(gstate, torch.tensor([action], dtype=torch.int64))
            if bool(done[0]):
                break
    print(f"\ncompared {total_plies} plies across {n_games} games; {mism} mismatches")
    print("per-plane max |CPU-GPU| diff:")
    names = ["P","N","B","R","Q","K","p","n","b","r","q","k",
             "castOK","castOQ","castEK","castEQ","ep","turn","movecnt","rep2","rep3","noprog"]
    for i in range(22):
        flag = "  <<<" if per_plane_maxdiff[i] > 1e-5 else ""
        print(f"  plane {i:2d} {names[i]:>8}: {per_plane_maxdiff[i]:.4f}{flag}")


if __name__ == "__main__":
    main()
