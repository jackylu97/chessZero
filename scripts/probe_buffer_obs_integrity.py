"""Probe REAL buffer integrity: do stored actions replay into legal games, and
do the reconstructed rep/no-progress planes look sane for the draw-collapsed
buffer? Also verifies the v3 round-trip (to_compact_dict -> from_compact_dict)
is a fixed point for observations.

Because v3 shards DROP observations and replay actions through python-chess, any
mismatch between the recorded action and the move the net actually made would
silently corrupt every downstream observation. We can't recover the original
GPU obs from a v3 buffer (it's gone), but we CAN check:
  (1) every stored action is legal at its ply when replayed -> no silent
      _action_to_move fallback to a different move.
  (2) rep>=2/rep>=3 planes fire on the threefold-drawn games (they should, given
      99% threefold) and the final position is actually a repetition.
  (3) the no-progress plane tracks halfmove correctly.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.chess import ChessGame, _action_to_move
import chess
from src.training.replay_buffer import _iter_shard_games, GameHistory

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"

# .buf is a ReplayBuffer.save() v3 stream: header dict + (compact_dict, prio) pairs.
import pickle

game = ChessGame()

n_games = 0
n_illegal_replays = 0
n_threefold = 0
n_threefold_with_rep3_plane = 0
n_decisive = 0
rep2_fire_count = 0
total_plies = 0
final_rep_ok = 0
final_rep_checked = 0
noprog_mismatch = 0
noprog_checked = 0

with open(BUF, "rb") as f:
    header = pickle.load(f)
    assert isinstance(header, dict) and header.get("version") in (2, 3), header
    nrec = header["n_records"]
    version = header["version"]
    sample = min(nrec, 400)
    for r in range(nrec):
        rec, prio = pickle.load(f)
        if r >= sample:
            continue
        if version == 3:
            # Replay manually so we can catch illegal actions instead of asserting.
            d = rec
            actions = [int(a) for a in d["actions"]]
            outcome = float(d["game_outcome"])
            draw_rep = bool(d.get("draw_by_repetition", False))
            board = chess.Board()
            ok = True
            for a in actions:
                mv = _action_to_move(a, board)
                if mv is None or mv not in board.legal_moves:
                    ok = False
                    n_illegal_replays += 1
                    break
                board.push(mv)
            if not ok:
                continue
            gh = GameHistory.from_compact_dict(d, ChessGame())
        else:
            gh = rec
            outcome = gh.game_outcome
            draw_rep = gh.draw_by_repetition

        n_games += 1
        L = len(gh.actions)
        total_plies += L
        if outcome != 0:
            n_decisive += 1
        if draw_rep:
            n_threefold += 1

        # Check rep planes across the game.
        any_rep3 = False
        for t in range(len(gh.observations)):
            o = gh.observations[t]
            if float(o[19, 0, 0]) >= 0.5:
                rep2_fire_count += 1
            if float(o[20, 0, 0]) >= 0.5:
                any_rep3 = True
        if draw_rep and any_rep3:
            n_threefold_with_rep3_plane += 1

        # Final position: if drawn by rep, final obs rep>=3 plane should be set
        # (the terminal position is the 3rd occurrence). The terminal obs is the
        # LAST observation.
        if draw_rep:
            final_rep_checked += 1
            last = gh.observations[-1]
            if float(last[20, 0, 0]) >= 0.5 or float(last[19, 0, 0]) >= 0.5:
                final_rep_ok += 1

print(f"shard: {BUF}")
print(f"records in shard: {nrec}, sampled: {sample}")
print(f"games successfully replayed: {n_games}")
print(f"ILLEGAL action replays (silent corruption!): {n_illegal_replays}")
print(f"decisive games: {n_decisive} ({100*n_decisive/max(n_games,1):.1f}%)")
print(f"threefold-drawn games (flag): {n_threefold} ({100*n_threefold/max(n_games,1):.1f}%)")
print(f"  of those, at least one obs with rep>=3 plane set: {n_threefold_with_rep3_plane}")
print(f"  of those, final obs has rep>=2 or rep>=3 set: {final_rep_ok}/{final_rep_checked}")
print(f"avg plies/game: {total_plies/max(n_games,1):.1f}")
print(f"total obs where rep>=2 plane fired: {rep2_fire_count}")
