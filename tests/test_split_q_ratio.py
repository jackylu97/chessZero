"""Split q_ratio: warmstart vs self-play phases use independent blend weights.

Verifies _wdl_target_at picks warmstart_q_ratio for positions with
external_values (Stockfish teacher) and selfplay_q_ratio otherwise, and that
both fall back to the single q_ratio when unset (back-compat).
"""
from __future__ import annotations

import numpy as np
import torch

from src.training.replay_buffer import GameHistory
from src.model.utils import eval_to_wdl

A = 9  # action space (tictactoe-ish; irrelevant to value target)


def _wdl(scalar):
    return np.array(eval_to_wdl(scalar, alpha=4.0, beta=2.0), dtype=np.float32)


def _warmstart_game(n=3, outcome=1.0, evals=(0.6, -0.4, 0.2)):
    g = GameHistory(game_name="tictactoe")
    g.game_outcome = outcome
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


def _selfplay_game(n=3, outcome=-1.0, rvs=(0.3, -0.5, 0.1)):
    g = GameHistory(game_name="tictactoe")
    g.game_outcome = outcome
    for i in range(n):
        g.observations.append(torch.zeros(1, 3, 3))
        g.actions.append(i % A)
        g.policies.append(np.full(A, 1.0 / A, dtype=np.float32))
        g.root_values.append(float(rvs[i]))
        g.legal_actions_list.append(list(range(A)))
        g.rewards.append(0.0)
    g.observations.append(torch.zeros(1, 3, 3))
    return g


def _value0(game, **kw):
    _, _, _, values, _, _ = game.make_target(
        0, 0, 5, 0.997, A, value_head_type="wdl", history_frames=1, **kw)
    return np.asarray(values[0], dtype=np.float32)


def test_warmstart_uses_warmstart_q():
    g = _warmstart_game(outcome=1.0, evals=(0.6, 0, 0))
    # ply0: white to move, outcome +1 → stm won → [1,0,0]
    expect = 0.5 * _wdl(0.6) + 0.5 * np.array([1, 0, 0], np.float32)
    got = _value0(g, warmstart_q_ratio=0.5, selfplay_q_ratio=0.1)
    assert np.allclose(got, expect, atol=1e-5), (got, expect)


def test_selfplay_uses_selfplay_q():
    g = _selfplay_game(outcome=-1.0, rvs=(0.3, 0, 0))
    # ply0: white to move, outcome -1 → stm lost → [0,0,1]
    expect = 0.9 * np.array([0, 0, 1], np.float32) + 0.1 * _wdl(0.3)
    got = _value0(g, warmstart_q_ratio=0.5, selfplay_q_ratio=0.1)
    assert np.allclose(got, expect, atol=1e-5), (got, expect)


def test_phases_are_independent():
    """selfplay_q_ratio must NOT affect a warmstart position and vice versa."""
    gw = _warmstart_game(outcome=1.0, evals=(0.6, 0, 0))
    # vary only selfplay_q_ratio → warmstart target unchanged
    a = _value0(gw, warmstart_q_ratio=0.5, selfplay_q_ratio=0.1)
    b = _value0(gw, warmstart_q_ratio=0.5, selfplay_q_ratio=0.9)
    assert np.allclose(a, b, atol=1e-6)


def test_fallback_to_single_q_ratio():
    """When split knobs are None, both phases use the single q_ratio."""
    gw = _warmstart_game(outcome=1.0, evals=(0.6, 0, 0))
    split = _value0(gw, q_ratio=0.0, warmstart_q_ratio=0.3, selfplay_q_ratio=0.9)
    single = _value0(gw, q_ratio=0.3)  # split knobs default None → fall back
    assert np.allclose(split, single, atol=1e-6)


def test_zero_q_is_pure_legacy():
    gw = _warmstart_game(outcome=1.0, evals=(0.6, 0, 0))
    got = _value0(gw, warmstart_q_ratio=0.0, selfplay_q_ratio=0.0)
    assert np.allclose(got, _wdl(0.6), atol=1e-5)  # pure eval_to_wdl, no outcome
