"""Tests for AlphaZero-style T-frame history encoding."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.replay_buffer import GameHistory, stack_with_history
from src.games.chess import ChessGame


# --- stack_with_history (used at inference time) ---------------------------

def test_stack_with_history_n1_returns_current():
    """n_frames=1 → no allocation, returns current_obs unchanged."""
    cur = torch.randn(3, 4, 4)
    out = stack_with_history(cur, [], n_frames=1)
    assert out is cur


def test_stack_with_history_zero_pad_at_game_start():
    """Empty prior list → newest frame + (T-1) zero frames."""
    cur = torch.ones(3, 4, 4)
    out = stack_with_history(cur, [], n_frames=4)
    assert out.shape == (12, 4, 4)
    # First 3 channels are cur (newest at front).
    assert torch.all(out[0:3] == 1.0)
    # Remaining channels zero-padded.
    assert torch.all(out[3:] == 0.0)


def test_stack_with_history_full_window():
    """T-1 prior frames available → cur + last T-1 priors, no padding."""
    prior = [torch.full((3, 4, 4), float(i)) for i in range(5)]  # 5 prior frames
    cur = torch.full((3, 4, 4), 99.0)
    out = stack_with_history(cur, prior, n_frames=3)
    assert out.shape == (9, 4, 4)
    # newest first: cur, prior[-1], prior[-2]
    assert torch.all(out[0:3] == 99.0)
    assert torch.all(out[3:6] == 4.0)
    assert torch.all(out[6:9] == 3.0)


def test_stack_with_history_partial_pad():
    """Fewer prior frames than T-1 → fill with zeros at end."""
    prior = [torch.full((3, 4, 4), 7.0)]  # 1 prior frame
    cur = torch.full((3, 4, 4), 99.0)
    out = stack_with_history(cur, prior, n_frames=4)
    assert out.shape == (12, 4, 4)
    # newest first: cur, prior[-1], 0, 0
    assert torch.all(out[0:3] == 99.0)
    assert torch.all(out[3:6] == 7.0)
    assert torch.all(out[6:9] == 0.0)
    assert torch.all(out[9:12] == 0.0)


# --- GameHistory._stack_history (used at training sample time) ------------

def _build_chess_history(n_plies: int = 6) -> GameHistory:
    """Synthesize a chess GameHistory with deterministic move-numbered observations.

    Each observation is filled with the ply index for easy assertion.
    """
    game = ChessGame()
    state = game.reset()
    gh = GameHistory(game_name="chess")
    # Replace each obs with a marker (but keep shape correct)
    obs0 = game.to_tensor(state)
    gh.observations.append(torch.full_like(obs0, 0.0))
    for ply in range(1, n_plies + 1):
        legal = game.legal_actions(state)
        a = legal[0]
        p = np.zeros(game.action_space_size, dtype=np.float32); p[a] = 1.0
        gh.actions.append(a); gh.policies.append(p)
        gh.root_values.append(0.0); gh.legal_actions_list.append(legal)
        state, r, _ = game.step(state, a)
        gh.rewards.append(r)
        gh.observations.append(torch.full_like(obs0, float(ply)))
    return gh


def test_gamehistory_stack_history_n1_returns_single_obs():
    gh = _build_chess_history(n_plies=4)
    out = gh._stack_history(idx=2, n_frames=1)
    # Should be obs[2] (which we filled with float 2.0)
    assert torch.all(out == 2.0)


def test_gamehistory_stack_history_full_window_no_pad():
    gh = _build_chess_history(n_plies=10)
    # idx=5 with T=4 → newest first: obs[5], obs[4], obs[3], obs[2]
    out = gh._stack_history(idx=5, n_frames=4)
    n_planes = ChessGame().num_planes
    assert out.shape == (n_planes * 4, 8, 8)
    assert torch.all(out[0:n_planes] == 5.0)
    assert torch.all(out[n_planes:2*n_planes] == 4.0)
    assert torch.all(out[2*n_planes:3*n_planes] == 3.0)
    assert torch.all(out[3*n_planes:4*n_planes] == 2.0)


def test_gamehistory_stack_history_partial_pad():
    gh = _build_chess_history(n_plies=10)
    # idx=2 with T=8 → newest=obs[2], then obs[1], obs[0], 0, 0, 0, 0, 0
    out = gh._stack_history(idx=2, n_frames=8)
    n = ChessGame().num_planes  # 22 (was 19 before repetition/no-progress planes)
    assert out.shape == (n * 8, 8, 8)
    assert torch.all(out[0:n] == 2.0)
    assert torch.all(out[n:2*n] == 1.0)
    assert torch.all(out[2*n:3*n] == 0.0)  # obs[0] is 0.0 by construction
    # Slots t=3..7 are zero-padded (no obs at those indices).
    for t in range(3, 8):
        assert torch.all(out[t*n:(t+1)*n] == 0.0)


# --- make_target with history_frames > 1 -----------------------------------

def test_make_target_history_8_target_obs_shape():
    gh = _build_chess_history(n_plies=20)
    obs, _, _, _, _, _ = gh.make_target(
        state_index=10, num_unroll_steps=3,
        td_steps=-1, discount=1.0,
        action_space_size=ChessGame().action_space_size,
        history_frames=8,
    )
    n = ChessGame().num_planes  # 22 (was 19 before repetition/no-progress planes)
    # K+1=4 observations, each shape (n*8, 8, 8)
    assert len(obs) == 4
    for o in obs:
        assert o.shape == (n * 8, 8, 8)


def test_make_target_history_window_correct_at_root():
    """At idx=10 with T=4, the root obs should contain plies 10, 9, 8, 7."""
    gh = _build_chess_history(n_plies=20)
    obs, _, _, _, _, _ = gh.make_target(
        state_index=10, num_unroll_steps=0,  # only root
        td_steps=-1, discount=1.0,
        action_space_size=ChessGame().action_space_size,
        history_frames=4,
    )
    root_stack = obs[0]
    n = ChessGame().num_planes  # 22 (was 19 before repetition/no-progress planes)
    assert torch.all(root_stack[0:n] == 10.0)
    assert torch.all(root_stack[n:2*n] == 9.0)
    assert torch.all(root_stack[2*n:3*n] == 8.0)
    assert torch.all(root_stack[3*n:4*n] == 7.0)


def test_make_target_default_history_unchanged():
    """history_frames=1 (default) returns single-frame observations
    (no stacking, back-compat with pre-history pipelines)."""
    gh = _build_chess_history(n_plies=10)
    obs, _, _, _, _, _ = gh.make_target(
        state_index=5, num_unroll_steps=2,
        td_steps=-1, discount=1.0,
        action_space_size=ChessGame().action_space_size,
        # history_frames defaults to 1
    )
    n = ChessGame().num_planes  # 22 (was 19 before repetition/no-progress planes)
    for o in obs:
        # Single frame: shape (n, 8, 8), not (n*T, 8, 8)
        assert o.shape == (n, 8, 8)


# --- Plane order convention ------------------------------------------------

def test_plane_order_newest_first_consistency():
    """Verify both stack_with_history (inference) and GameHistory._stack_history
    (training) put the newest frame at channel 0."""
    gh = _build_chess_history(n_plies=8)

    # stack_with_history at inference
    cur = torch.full_like(gh.observations[0], 99.0)
    inference_stack = stack_with_history(cur, list(gh.observations), n_frames=4)
    n = ChessGame().num_planes  # 22 (was 19 before repetition/no-progress planes)
    assert torch.all(inference_stack[0:n] == 99.0)  # cur is newest
    assert torch.all(inference_stack[n:2*n] == 8.0)  # then most recent prior

    # GameHistory._stack_history at training
    training_stack = gh._stack_history(idx=8, n_frames=4)
    assert torch.all(training_stack[0:n] == 8.0)  # idx=8 is newest
    assert torch.all(training_stack[n:2*n] == 7.0)
