"""Threefold-repetition value-target penalty.

A self-play game that ended in threefold repetition (draw_by_repetition=True)
has its draw value target tilted from [0, 1, 0] to [0, 1-δ, δ] — moving δ mass
from Draw to Loss — STM-symmetric (same tilt for white-to-move and
black-to-move plies). Non-repetition draws, decisive games, and warmstart
(external_values) games are unaffected. δ=0 reproduces legacy behavior.
"""
from __future__ import annotations

import numpy as np
import torch

from src.training.replay_buffer import GameHistory
from src.model.utils import eval_to_wdl

A = 9  # action space (irrelevant to the value target)

DRAW = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _draw_selfplay_game(n=4, draw_by_repetition=False):
    """Self-play game (no external_values) that drew (game_outcome=0)."""
    g = GameHistory(game_name="tictactoe")
    g.game_outcome = 0.0
    g.draw_by_repetition = draw_by_repetition
    for i in range(n):
        g.observations.append(torch.zeros(1, 3, 3))
        g.actions.append(i % A)
        g.policies.append(np.full(A, 1.0 / A, dtype=np.float32))
        g.root_values.append(0.0)
        g.legal_actions_list.append(list(range(A)))
        g.rewards.append(0.0)
    g.observations.append(torch.zeros(1, 3, 3))
    return g


def _warmstart_game(n=4, evals=(0.2, -0.2, 0.1, 0.0)):
    """Warmstart game (external_values present), drew, marked repetition."""
    g = GameHistory(game_name="tictactoe")
    g.game_outcome = 0.0
    g.draw_by_repetition = True
    for i in range(n):
        g.observations.append(torch.zeros(1, 3, 3))
        g.actions.append(i % A)
        g.policies.append(np.full(A, 1.0 / A, dtype=np.float32))
        g.root_values.append(0.0)
        g.external_values.append(float(evals[i]))
        g.legal_actions_list.append(list(range(A)))
        g.rewards.append(0.0)
    g.observations.append(torch.zeros(1, 3, 3))
    return g


def _value_at(game, ply, **kw):
    """Value target at `ply` via make_target rooted at that ply."""
    _, _, _, values, _, _ = game.make_target(
        ply, 0, 5, 0.997, A, value_head_type="wdl", history_frames=1, **kw)
    return np.asarray(values[0], dtype=np.float32)


def test_repetition_draw_tilts_toward_loss_both_stm():
    g = _draw_selfplay_game(n=4, draw_by_repetition=True)
    tilted = np.array([0.0, 0.8, 0.2], dtype=np.float32)
    # ply 0 (white to move) and ply 1 (black to move) — STM-symmetric.
    for ply in (0, 1, 2, 3):
        v = _value_at(g, ply, repetition_penalty=0.2, selfplay_q_ratio=0.0)
        np.testing.assert_allclose(v, tilted, atol=1e-6,
                                   err_msg=f"ply {ply} not loss-tilted")


def test_non_repetition_draw_unaffected():
    g = _draw_selfplay_game(n=4, draw_by_repetition=False)
    for ply in (0, 1):
        v = _value_at(g, ply, repetition_penalty=0.2, selfplay_q_ratio=0.0)
        np.testing.assert_allclose(v, DRAW, atol=1e-6)


def test_delta_zero_back_compat():
    g = _draw_selfplay_game(n=4, draw_by_repetition=True)
    for ply in (0, 1):
        v = _value_at(g, ply, repetition_penalty=0.0, selfplay_q_ratio=0.0)
        np.testing.assert_allclose(v, DRAW, atol=1e-6)


def test_warmstart_game_unaffected_by_penalty():
    g = _warmstart_game(n=4, evals=(0.2, -0.2, 0.1, 0.0))
    for ply in range(4):
        base = _value_at(g, ply, repetition_penalty=0.0, warmstart_q_ratio=0.0)
        with_pen = _value_at(g, ply, repetition_penalty=0.5, warmstart_q_ratio=0.0)
        # Warmstart legacy target is eval_to_wdl(external), penalty must not move it.
        expect = np.array(eval_to_wdl(g.external_values[ply], alpha=4.0, beta=2.0),
                          dtype=np.float32)
        np.testing.assert_allclose(base, expect, atol=1e-6)
        np.testing.assert_allclose(with_pen, base, atol=1e-6)
