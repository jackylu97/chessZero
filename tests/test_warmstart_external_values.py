"""Tests for the Stockfish warmstart data-layer path.

Verifies that GameHistory.external_values, when populated, takes precedence
over the td_steps/game_outcome value-target branches in make_target.
"""

import pickle

import numpy as np
import pytest
import torch

from src.training.replay_buffer import GameHistory, ReplayBuffer


ACTION_SPACE = 64
NUM_UNROLL = 3
DISCOUNT = 0.997


def _synthetic_game(n_moves: int, external_values: list[float] | None = None) -> GameHistory:
    g = GameHistory(game_name="chess")
    for i in range(n_moves):
        g.observations.append(torch.zeros(3, 4, 4))
        g.actions.append(i % ACTION_SPACE)
        g.rewards.append(0.0)
        g.policies.append(np.full(ACTION_SPACE, 1.0 / ACTION_SPACE, dtype=np.float32))
        g.root_values.append(0.0)
        g.legal_actions_list.append([0, 1, 2])
    # Terminal observation (self-play convention: N+1 observations, N everything else).
    g.observations.append(torch.zeros(3, 4, 4))
    g.game_outcome = 1.0  # white wins
    if external_values is not None:
        g.external_values = list(external_values)
    return g


def test_make_target_uses_external_values_when_populated():
    ext = [0.10, -0.20, 0.30, -0.40, 0.50, -0.60]  # 6 entries for 5 moves + terminal
    g = _synthetic_game(n_moves=5, external_values=ext)

    _, _, _, values, _, _ = g.make_target(
        state_index=0,
        num_unroll_steps=NUM_UNROLL,
        td_steps=-1,
        discount=DISCOUNT,
        action_space_size=ACTION_SPACE,
    )

    assert values[:4] == pytest.approx(ext[:4])


def test_make_target_falls_back_to_game_outcome_when_external_empty():
    g = _synthetic_game(n_moves=5, external_values=None)

    _, _, _, values, _, _ = g.make_target(
        state_index=0,
        num_unroll_steps=NUM_UNROLL,
        td_steps=-1,
        discount=DISCOUNT,
        action_space_size=ACTION_SPACE,
    )

    # td_steps=-1, game_outcome=+1, sign alternates: +1, -1, +1, -1
    assert values == pytest.approx([1.0, -1.0, 1.0, -1.0])


def test_make_target_respects_external_values_partial():
    """If external_values runs out before the unroll window ends, later indices
    should still fall through to the td_steps branch (no IndexError, no silent
    truncation to zero)."""
    ext = [0.1, 0.2]  # only 2 entries for state_index=1 .. 1+3
    g = _synthetic_game(n_moves=5, external_values=ext)

    _, _, _, values, _, _ = g.make_target(
        state_index=1,
        num_unroll_steps=NUM_UNROLL,
        td_steps=-1,
        discount=DISCOUNT,
        action_space_size=ACTION_SPACE,
    )

    # idx=1 → ext[1]=0.2. idx=2,3,4 → no ext entry → td path: +1, -1, +1 (idx%2).
    assert values[0] == pytest.approx(0.2)
    assert values[1:] == pytest.approx([1.0, -1.0, 1.0])


def test_load_warmstart_games_round_trips(tmp_path):
    ext = [0.5, -0.5, 0.5]
    games = [_synthetic_game(n_moves=2, external_values=ext) for _ in range(4)]
    shard_path = tmp_path / "shard_000.pkl"
    with open(shard_path, "wb") as f:
        pickle.dump(games, f)

    buf = ReplayBuffer(max_size=10)
    loaded = buf.load_warmstart_games([shard_path])
    assert loaded == 4
    assert len(buf) == 4
    # Round-tripped external_values preserved
    assert buf.buffer[0].external_values == pytest.approx(ext)


def test_load_warmstart_games_rejects_bad_shard(tmp_path):
    shard_path = tmp_path / "bad.pkl"
    with open(shard_path, "wb") as f:
        pickle.dump({"not": "a list"}, f)
    buf = ReplayBuffer(max_size=10)
    with pytest.raises(TypeError):
        buf.load_warmstart_games([shard_path])
