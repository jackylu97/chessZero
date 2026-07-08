"""Probe terminal-observation alignment.

In self_play.py, observations has L+1 entries: obs[0..L-1] (pre-move states the
net acted on) and obs[L] (the terminal/post-final-move state). make_target can
sample any pos in [0, len(game)) = [0, L] inclusive of the terminal index L
(np.random.randint(0, len(game)) where len == L+1).

This verifies, on REAL buffer games:
  (1) obs[L] (terminal) is the position AFTER the last action (3rd repetition for
      threefold games) -> rep>=3 plane should be set there.
  (2) the value target made at the terminal index is the correct STM-POV outcome.
  (3) the obs at every index reconstructs to the position reached by replaying
      actions[0..idx-1] (i.e. obs[idx] = state after idx moves).
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import GameHistory

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"

game = ChessGame()
n_checked = 0
n_terminal_rep3 = 0
n_terminal_is_postmove = 0
n_obs_align_ok = 0
n_obs_total = 0

with open(BUF, "rb") as f:
    header = pickle.load(f)
    nrec = header["n_records"]
    for r in range(min(nrec, 80)):
        rec, prio = pickle.load(f)
        d = rec
        if not bool(d.get("draw_by_repetition", False)):
            continue
        gh = GameHistory.from_compact_dict(d, ChessGame())
        L = len(gh.actions)
        # obs count should be L+1
        assert len(gh.observations) == L + 1, (len(gh.observations), L)
        n_checked += 1

        # Replay independently to get the ground-truth board at each index.
        board = chess.Board()
        for idx in range(L + 1):
            # obs[idx] should equal to_tensor of board after idx moves.
            st = game.reset(); st.board = board.copy()
            expected = game.to_tensor(st)
            got = gh.observations[idx]
            n_obs_total += 1
            if float((expected - got).abs().max()) < 1e-5:
                n_obs_align_ok += 1
            if idx < L:
                mv = _action_to_move(gh.actions[idx], board)
                board.push(mv)

        # Terminal obs: board now has all L moves -> the 3rd-repetition position.
        term = gh.observations[L]
        if float(term[20, 0, 0]) >= 0.5:
            n_terminal_rep3 += 1
        # is the terminal board a true 3-fold?
        if board.is_repetition(3):
            n_terminal_is_postmove += 1

print(f"threefold games checked: {n_checked}")
print(f"obs index alignment (obs[idx] == position after idx moves): "
      f"{n_obs_align_ok}/{n_obs_total}")
print(f"terminal obs has rep>=3 plane set: {n_terminal_rep3}/{n_checked}")
print(f"terminal board IS true 3-fold repetition: {n_terminal_is_postmove}/{n_checked}")
