"""Regression tests for the q_ratio value-target blend in GameHistory.make_target.

Definition under test (src/training/replay_buffer.py::_wdl_target_at):
  q = weight on the "blended-in" signal; (1-q) = weight on the legacy target.
  WARMSTART (external_values present at ply i):
    target = (1-q)·eval_to_wdl(external_values[i]) + q·outcome_onehot
  SELF-PLAY (no external_values for ply i):
    target = (1-q)·outcome_onehot + q·eval_to_wdl(root_values[i])
  q==0.0 MUST reduce to the legacy targets exactly (back-compat).
"""
import numpy as np
import torch

from src.training.replay_buffer import GameHistory
from src.model.utils import eval_to_wdl

A, B = 4.0, 2.0  # eval_to_wdl alpha, beta
N = 8            # plies


def _obs():
    return torch.zeros(1, 8, 8)


def _onehot(outcome, ply):
    """STM-relative one-hot of the game outcome at ply."""
    if outcome == 0.0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    stm_is_white = (ply % 2 == 0)
    stm_won = (outcome > 0.0) == stm_is_white
    return (np.array([1.0, 0.0, 0.0], dtype=np.float32) if stm_won
            else np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _evwdl(x):
    return np.array(eval_to_wdl(x, alpha=A, beta=B), dtype=np.float32)


def _make_warmstart():
    """Warmstart game: external_values populated, decisive outcome (white wins)."""
    gh = GameHistory()
    gh.observations = [_obs() for _ in range(N + 1)]
    gh.actions = [0] * N
    gh.rewards = [0.0] * N
    gh.policies = [np.zeros(64, dtype=np.float32) for _ in range(N)]
    gh.root_values = [0.0] * N
    gh.game_outcome = 1.0
    # Mildly graded STM-POV evals (not saturating) so blend differs from legacy.
    gh.external_values = [0.3, -0.2, 0.5, -0.1, 0.4, 0.0, 0.6, -0.3]
    return gh


def _make_selfplay():
    """Self-play game: no external_values, nonzero root_values, decisive (black wins)."""
    gh = GameHistory()
    gh.observations = [_obs() for _ in range(N + 1)]
    gh.actions = [0] * N
    gh.rewards = [0.0] * N
    gh.policies = [np.zeros(64, dtype=np.float32) for _ in range(N)]
    gh.root_values = [0.25, -0.4, 0.1, -0.5, 0.3, -0.2, 0.45, -0.15]
    gh.game_outcome = -1.0
    gh.external_values = []
    return gh


def _target_at(gh, ply, q):
    """Return the in-range WDL value target at a single ply via make_target."""
    _, _, _, values, _, _ = gh.make_target(
        ply, 0, -1, 0.997, 64,
        value_head_type="wdl", history_frames=1,
        eval_to_wdl_alpha=A, eval_to_wdl_beta=B, q_ratio=q,
    )
    return np.asarray(values[0], dtype=np.float32)


# ---------------------------------------------------------------------------

def test_qratio_zero_is_legacy_warmstart():
    gh = _make_warmstart()
    for ply in range(N):
        got = _target_at(gh, ply, 0.0)
        legacy = _evwdl(gh.external_values[ply])
        assert np.allclose(got, legacy, atol=1e-6), (ply, got, legacy)


def test_qratio_zero_is_legacy_selfplay():
    gh = _make_selfplay()
    for ply in range(N):
        got = _target_at(gh, ply, 0.0)
        legacy = _onehot(gh.game_outcome, ply)
        assert np.array_equal(got, legacy), (ply, got, legacy)


def test_qratio_one_pure_blended_in_warmstart():
    # q=1 → warmstart target is pure outcome one-hot.
    gh = _make_warmstart()
    for ply in range(N):
        got = _target_at(gh, ply, 1.0)
        expect = _onehot(gh.game_outcome, ply)
        assert np.allclose(got, expect, atol=1e-6), (ply, got, expect)


def test_qratio_one_pure_blended_in_selfplay():
    # q=1 → self-play target is pure eval_to_wdl(root_value).
    gh = _make_selfplay()
    for ply in range(N):
        got = _target_at(gh, ply, 1.0)
        expect = _evwdl(gh.root_values[ply])
        assert np.allclose(got, expect, atol=1e-6), (ply, got, expect)


def test_qratio_half_mixture_warmstart():
    gh = _make_warmstart()
    for ply in range(N):
        got = _target_at(gh, ply, 0.5)
        legacy = _evwdl(gh.external_values[ply])
        blend = _onehot(gh.game_outcome, ply)
        expect = 0.5 * legacy + 0.5 * blend
        assert np.allclose(got, expect, atol=1e-6), (ply, got, expect)
        assert abs(float(got.sum()) - 1.0) < 1e-6
        assert (got >= 0.0).all() and (got <= 1.0).all()


def test_qratio_half_mixture_selfplay():
    gh = _make_selfplay()
    for ply in range(N):
        got = _target_at(gh, ply, 0.5)
        legacy = _onehot(gh.game_outcome, ply)
        blend = _evwdl(gh.root_values[ply])
        expect = 0.5 * legacy + 0.5 * blend
        assert np.allclose(got, expect, atol=1e-6), (ply, got, expect)
        assert abs(float(got.sum()) - 1.0) < 1e-6
        assert (got >= 0.0).all() and (got <= 1.0).all()


def test_stm_parity():
    # White-win warmstart: even plies (STM=white) → won one-hot under q=1,
    # odd plies (STM=black) → lost one-hot.
    gh = _make_warmstart()
    win = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    loss = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for ply in range(N):
        got = _target_at(gh, ply, 1.0)
        expect = win if ply % 2 == 0 else loss
        assert np.array_equal(got, expect), (ply, got, expect)


def test_selfplay_root_values_short_falls_back_to_outcome():
    # If root_values is shorter than the ply, q>0 self-play falls back to outcome.
    gh = _make_selfplay()
    gh.root_values = [0.25, -0.4]  # only 2 entries
    for ply in range(2, N):
        got = _target_at(gh, ply, 0.7)
        expect = _onehot(gh.game_outcome, ply)
        assert np.array_equal(got, expect), (ply, got, expect)
