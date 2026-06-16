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


def _draw_selfplay_game(n=4, draw_by_repetition=False, draw_by_no_progress=False):
    """Self-play game (no external_values) that drew (game_outcome=0)."""
    g = GameHistory(game_name="tictactoe")
    g.game_outcome = 0.0
    g.draw_by_repetition = draw_by_repetition
    g.draw_by_no_progress = draw_by_no_progress
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


def test_no_progress_draw_tilts_toward_loss():
    """75-move / no-progress draws get the same loss-tilt as threefold."""
    g = _draw_selfplay_game(n=4, draw_by_no_progress=True)
    tilted = np.array([0.0, 0.8, 0.2], dtype=np.float32)
    for ply in (0, 1, 2, 3):
        v = _value_at(g, ply, repetition_penalty=0.2, selfplay_q_ratio=0.0)
        np.testing.assert_allclose(v, tilted, atol=1e-6,
                                   err_msg=f"no-progress ply {ply} not loss-tilted")


def test_neither_flag_draw_unaffected():
    """A draw that is neither threefold nor no-progress (stalemate / insufficient
    material / ply-cap) keeps the pure draw target."""
    g = _draw_selfplay_game(n=4, draw_by_repetition=False, draw_by_no_progress=False)
    for ply in (0, 1):
        v = _value_at(g, ply, repetition_penalty=0.2, selfplay_q_ratio=0.0)
        np.testing.assert_allclose(v, DRAW, atol=1e-6)


def test_delta_zero_back_compat():
    g = _draw_selfplay_game(n=4, draw_by_repetition=True)
    for ply in (0, 1):
        v = _value_at(g, ply, repetition_penalty=0.0, selfplay_q_ratio=0.0)
        np.testing.assert_allclose(v, DRAW, atol=1e-6)


def test_per_ply_window_ramps_delta_toward_the_draw():
    """window>0: full δ at the terminal drawn position, linearly →0 `window`
    plies before it. Game length L=len(observations); terminal at L-1."""
    g = _draw_selfplay_game(n=6, draw_by_repetition=True)  # 6 plies + terminal obs → L=7
    L = len(g)
    assert L == 7
    end = L - 1  # terminal index 6
    window = 4
    delta = 0.4
    # ply p ⇒ weight = max(0, 1 - (end-p)/window); δ_p = delta*weight
    for p in range(L):
        plies_to_end = end - p
        w = max(0.0, 1.0 - plies_to_end / window)
        expect_d = delta * w
        v = _value_at(g, p, repetition_penalty=delta,
                      repetition_penalty_window=window, selfplay_q_ratio=0.0)
        np.testing.assert_allclose(
            v, [0.0, 1.0 - expect_d, expect_d], atol=1e-6,
            err_msg=f"ply {p}: plies_to_end={plies_to_end} weight={w}")
    # Spot-check the shape: terminal ply gets full δ, the window-distant ply ~0.
    v_end = _value_at(g, end, repetition_penalty=delta, repetition_penalty_window=window,
                      selfplay_q_ratio=0.0)
    np.testing.assert_allclose(v_end, [0.0, 0.6, 0.4], atol=1e-6)  # full δ at draw
    v_far = _value_at(g, end - window, repetition_penalty=delta,
                      repetition_penalty_window=window, selfplay_q_ratio=0.0)
    np.testing.assert_allclose(v_far, DRAW, atol=1e-6)  # window plies back → no tilt


def test_window_zero_is_uniform_legacy():
    """window=0 reproduces the uniform full-δ tilt on every ply."""
    g = _draw_selfplay_game(n=5, draw_by_repetition=True)
    for p in range(5):
        v = _value_at(g, p, repetition_penalty=0.3, repetition_penalty_window=0,
                      selfplay_q_ratio=0.0)
        np.testing.assert_allclose(v, [0.0, 0.7, 0.3], atol=1e-6)


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
