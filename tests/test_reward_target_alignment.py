"""Reward-target alignment in GameHistory.make_target.

The trainer reads target_rewards[:, k+1] at unroll step k, where actions[:, k]
is self.actions[state_index + k] and recurrent_inference predicts the reward
of transition s_{state_index+k} → s_{state_index+k+1} = self.rewards[state_index + k].

For the loss to compare apples to apples, make_target must produce
target_rewards[k+1] == self.rewards[state_index + k] — i.e. the DeepMind
MuZero convention where target_rewards[i] is the reward INTO state
(state_index + i). target_rewards[0] is therefore the reward INTO the root
and unused by the trainer.
"""

import numpy as np
import pytest
import torch

from src.training.replay_buffer import GameHistory


ACTION_SPACE = 4


def _game_with_rewards(rewards: list[float], game_outcome: float = 0.0) -> GameHistory:
    """Synthetic GameHistory with N moves matching the given reward sequence.

    Mirrors the self_play convention: observations has N+1 entries (final
    terminal observation), every other list has N entries. self.rewards[t]
    is the reward of action t (transition s_t → s_{t+1}).
    """
    g = GameHistory(game_name="chess")
    n = len(rewards)
    for t in range(n):
        g.observations.append(torch.zeros(3, 4, 4))
        g.actions.append(t % ACTION_SPACE)
        g.rewards.append(float(rewards[t]))
        g.policies.append(np.full(ACTION_SPACE, 1.0 / ACTION_SPACE, dtype=np.float32))
        g.root_values.append(0.0)
        g.legal_actions_list.append(list(range(ACTION_SPACE)))
    g.observations.append(torch.zeros(3, 4, 4))  # terminal obs
    g.game_outcome = game_outcome
    return g


def _make_target_rewards(g: GameHistory, state_index: int, K: int) -> list[float]:
    _, _, _, _, rewards, _ = g.make_target(
        state_index=state_index,
        num_unroll_steps=K,
        td_steps=-1,
        discount=1.0,
        action_space_size=ACTION_SPACE,
    )
    return [float(r) for r in rewards]


def test_target_rewards_use_into_state_convention_at_root():
    """target_rewards[0] is the reward INTO the root (unused), target_rewards[i]
    for i >= 1 is the reward of transition s_{idx-1} → s_idx."""
    g = _game_with_rewards([0.0, 0.0, 0.0, 1.0])  # white mates on move 4
    out = _make_target_rewards(g, state_index=0, K=4)
    # target_rewards[0]: reward INTO root (state 0). No prior transition → 0.
    # target_rewards[k+1] for k=0..3: rewards of self.actions[0..3] = self.rewards[0..3].
    assert out == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])


def test_target_rewards_align_with_trainer_indexing():
    """At unroll step k the trainer reads target_rewards[k+1]; this must equal
    self.rewards[state_index + k] (the reward of the action being applied)."""
    g = _game_with_rewards([0.1, -0.2, 0.3, -0.4, 0.5])
    state_index = 1
    K = 3
    out = _make_target_rewards(g, state_index=state_index, K=K)
    for k in range(K):
        assert out[k + 1] == pytest.approx(g.rewards[state_index + k]), (
            f"target_rewards[{k+1}]={out[k+1]} != self.rewards[{state_index+k}]"
            f"={g.rewards[state_index+k]} — off-by-one regression"
        )


def test_target_rewards_terminal_within_window():
    """When state_index+K reaches the terminal index, the terminal reward
    must land on target_rewards[K] (the last unroll slot's prediction),
    NOT shifted off the end."""
    rewards = [0.0, 0.0, 0.0, 1.0]  # 4 moves, terminal reward on move 4
    g = _game_with_rewards(rewards, game_outcome=1.0)
    # Sample state_index=2 with K=2: action at k=1 is self.actions[3] (the
    # mating move), so the trainer at k=1 reads target_rewards[2] which must
    # be self.rewards[3] = 1.0.
    out = _make_target_rewards(g, state_index=2, K=2)
    assert out[2] == pytest.approx(1.0), (
        f"terminal reward not on target_rewards[K]; got {out}"
    )


def test_target_rewards_past_game_end_zero():
    """Past game end (idx beyond terminal observation), target_rewards is 0."""
    g = _game_with_rewards([0.0, 0.0, 0.5])  # 3 moves, len(self) = 4
    # state_index=2, K=4 → unroll touches indices 2, 3, 4, 5, 6.
    # idx=2,3 are valid; idx=4..6 are past-end → reward 0.
    out = _make_target_rewards(g, state_index=2, K=4)
    # idx=2: prev=1, reward = self.rewards[1] = 0.0
    # idx=3: prev=2, reward = self.rewards[2] = 0.5 (terminal)
    # idx=4..6: past end → 0
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2:] == [pytest.approx(0.0)] * 3


def test_target_rewards_at_state_index_zero_first_slot_is_zero():
    """target_rewards[0] (reward INTO root) is always 0 — there is no
    transition into s_0."""
    g = _game_with_rewards([0.7, -0.3])
    out = _make_target_rewards(g, state_index=0, K=2)
    assert out[0] == pytest.approx(0.0)
    # And the rest align.
    assert out[1] == pytest.approx(0.7)
    assert out[2] == pytest.approx(-0.3)
