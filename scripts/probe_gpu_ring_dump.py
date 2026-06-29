"""Dump the GPU rep-ring around the undercount to find the exact off-by-one.
Compare to the sequence of python-chess transposition keys (post-last-irreversible).
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import chess

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.games.chess_gpu import GpuChessGame, _zobrist_hash, REP_HISTORY_K
from src.training.replay_buffer import GameHistory


def get_game(path, game, idx):
    with open(path, "rb") as f:
        header = pickle.load(f)
        ver = header["version"]
        for i in range(header["n_records"]):
            record, prio = pickle.load(f)
            if i == idx:
                return GameHistory.from_compact_dict(record, game) if ver == 3 else record


def main():
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    bf = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
    gh = get_game(bf, game, 27)

    gpu = GpuChessGame(); gpu.max_plies = 1024
    gstate = gpu.reset_batch(1, device="cpu")
    board = chess.Board()

    # Track GPU ring AND hash of current position each ply
    for i, a in enumerate(gh.actions):
        at = torch.tensor([int(a)], dtype=torch.int64)
        gstate, _, _, _ = gpu.step_batch_with_legal(gstate, at)
        mv = _action_to_move(int(a), board)
        board.push(mv)
        if i + 1 in (28, 29, 30, 31, 32, 33, 34):
            cur = int(_zobrist_hash(gstate)[0].item())
            rc = int(gstate.rep_count[0].item())
            ring = gstate.rep_hashes[0, :rc].tolist()
            n_match = sum(1 for h in ring if h == cur)
            print(f"ply {i+1}: halfmove={board.halfmove_clock} rep_count={rc} "
                  f"cur_hash={cur}")
            print(f"   ring(valid)={ring}")
            print(f"   matches_of_cur_in_ring={n_match}  (cur present? {cur in ring})")
            # python-chess: how many times has this position occurred since last irreversible
            print(f"   python-chess is_repetition(2)={board.is_repetition(2)} "
                  f"is_repetition(3)={board.is_repetition(3)}")


if __name__ == "__main__":
    main()
