"""Minimal repro: replay a known threefold game on the GPU engine and on
python-chess, printing at each ply the GPU ring's occurrence-count of the
current position vs python-chess's true occurrence count. Isolates whether the
GPU UNDERCOUNTS repetitions (a generation-side bug) independent of obs/reload.
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import chess

from src.config import get_config
from src.games.chess import ChessGame
from src.games.chess_gpu import (
    GpuChessGame, _zobrist_hash, REP_HISTORY_K,
)
from src.training.replay_buffer import GameHistory


def get_game(path, game, idx):
    with open(path, "rb") as f:
        header = pickle.load(f)
        ver = header["version"]
        for i in range(header["n_records"]):
            record, prio = pickle.load(f)
            if i == idx:
                return GameHistory.from_compact_dict(record, game) if ver == 3 else record
    return None


def gpu_ring_count(state):
    """Number of times current position's hash appears in the valid ring."""
    cur = _zobrist_hash(state)
    if state.rep_hashes is None:
        return 1
    n = state.n
    rep_pos = torch.arange(REP_HISTORY_K).unsqueeze(0).expand(n, REP_HISTORY_K)
    valid = rep_pos < state.rep_count.to(torch.int64).unsqueeze(1)
    matches = (state.rep_hashes == cur.unsqueeze(1)) & valid
    return int(matches.sum(dim=1)[0].item())


def pc_count(board):
    """True occurrence count of the current position via python-chess transposition key."""
    # python-chess: count positions in move stack with same key as current.
    # Replicate is_repetition's notion: same transposition key (incl. castling, ep, side).
    cnt = 1
    key = board._transposition_key()
    # Walk back through the stack
    switchyard = []
    b = board.copy(stack=True)
    while b.move_stack:
        m = b.pop()
        switchyard.append(m)
        if b.is_irreversible(m):
            break
        if b._transposition_key() == key:
            cnt += 1
    return cnt


def main():
    cfg = get_config("chess_small")
    cfg.device = "cpu"
    game = ChessGame()
    bf = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
    gh = get_game(bf, game, 27)
    print(f"game 27 len={len(gh.actions)}")

    gpu = GpuChessGame(); gpu.max_plies = 1024
    gstate = gpu.reset_batch(1, device="cpu")
    board = chess.Board()

    print(f"{'ply':>4} {'gpu_ring':>8} {'pc_count':>8} {'gpu_rep_cnt(reset?)':>20} {'halfmove':>8}")
    for i, a in enumerate(gh.actions):
        at = torch.tensor([int(a)], dtype=torch.int64)
        gstate, _, _, _ = gpu.step_batch_with_legal(gstate, at)
        # also push to python-chess via UCI from action
        from src.games.chess import _action_to_move
        mv = _action_to_move(int(a), board)
        board.push(mv)
        g = gpu_ring_count(gstate)
        p = pc_count(board)
        rc = int(gstate.rep_count[0].item())
        flag = "  <-- UNDERCOUNT" if g < p else ("  <-- OVERCOUNT" if g > p else "")
        if g != p or i + 1 in (30, 34):
            print(f"{i+1:>4} {g:>8} {p:>8} {rc:>20} {board.halfmove_clock:>8}{flag}")


if __name__ == "__main__":
    main()
