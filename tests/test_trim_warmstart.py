"""ReplayBuffer.trim_warmstart_to — the two-phase boundary trim.

Phase 1 fills the buffer with warmstart games; at the self-play boundary the
warmstart pool is trimmed to the anchor size so the buffer can't balloon once
self-play games arrive. Verifies only the oldest warmstart games are dropped,
self-play games are untouched, and the keep-set is the most-recent warmstart.
"""
from __future__ import annotations

import numpy as np

from src.training.replay_buffer import GameHistory, ReplayBuffer


def _warm(tag: float) -> GameHistory:
    g = GameHistory(game_name="tictactoe")
    g.actions = [0]
    g.policies = [np.full(9, 1 / 9, dtype=np.float32)]
    g.root_values = [0.0]
    g.rewards = [0.0]
    g.external_values = [tag]   # presence of external_values ⇒ warmstart
    g.game_outcome = 0.0
    return g


def _selfplay(tag: float) -> GameHistory:
    g = GameHistory(game_name="tictactoe")
    g.actions = [0]
    g.policies = [np.full(9, 1 / 9, dtype=np.float32)]
    g.root_values = [tag]
    g.rewards = [0.0]
    g.external_values = []      # no external_values ⇒ self-play
    g.game_outcome = 0.0
    return g


def test_trim_keeps_most_recent_warmstart():
    buf = ReplayBuffer(max_size=1000)  # single-pool fill (Phase 1)
    for i in range(10):
        buf.save_game(_warm(float(i)))   # tags 0..9, oldest first
    dropped = buf.trim_warmstart_to(3)
    assert dropped == 7
    remaining = [g.external_values[0] for g in buf.buffer]
    assert remaining == [7.0, 8.0, 9.0]          # 3 most-recent kept, in order
    assert len(buf._priorities) == len(buf.buffer)  # priorities stay aligned


def test_trim_leaves_selfplay_untouched():
    buf = ReplayBuffer(max_size=1000)
    for i in range(5):
        buf.save_game(_warm(float(i)))
    for i in range(4):
        buf.save_game(_selfplay(float(100 + i)))
    dropped = buf.trim_warmstart_to(2)
    assert dropped == 3
    warm = [g.external_values[0] for g in buf.buffer if g.external_values]
    sp = [g.root_values[0] for g in buf.buffer if not g.external_values]
    assert warm == [3.0, 4.0]                       # 2 most-recent warmstart
    assert sp == [100.0, 101.0, 102.0, 103.0]       # all self-play kept


def test_trim_noop_when_under_target():
    buf = ReplayBuffer(max_size=1000)
    for i in range(3):
        buf.save_game(_warm(float(i)))
    assert buf.trim_warmstart_to(10) == 0
    assert len(buf.buffer) == 3


def test_no_balloon_after_trim_then_selfplay():
    """After trimming to the anchor and enabling two-pool, total stays bounded."""
    buf = ReplayBuffer(max_size=15)
    # Phase 1: fill with warmstart (single-pool cap 15).
    for i in range(15):
        buf.save_game(_warm(float(i)))
    assert len(buf.buffer) == 15
    # Boundary: trim to anchor 3, enable two-pool.
    buf.trim_warmstart_to(3)
    buf.warmstart_max_size = 3            # sp pool cap = 15 - 3 = 12
    assert len([g for g in buf.buffer if g.external_values]) == 3
    # Phase 2: stream self-play; buffer must not exceed max_size, anchor persists.
    for i in range(40):
        buf.save_game(_selfplay(float(i)))
    assert len(buf.buffer) <= buf.max_size
    assert len([g for g in buf.buffer if g.external_values]) == 3   # anchor held
