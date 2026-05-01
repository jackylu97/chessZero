"""TD n-step value-target construction in GameHistory.make_target.

The n-step return summed inside make_target adds per-ply rewards. In a
two-player zero-sum game (chess) self.rewards[j] is the reward of action j in
the mover-at-j (= STM-at-j) POV — that POV alternates white/black across
plies. To get a target in the STM-at-idx POV we must flip parity when j and
idx differ in oddness.

The bootstrap term (root_values[bootstrap_idx]) already does this parity
flip. The reward sum used to add rewards as-is, which mis-attributed any
non-zero terminal-near reward when its parity differed from the sample's.
This file pins the post-fix behavior.
"""

import numpy as np
import pytest
import torch

from src.training.replay_buffer import GameHistory


ACTION_SPACE = 4


def _game(rewards: list[float], root_values: list[float] | None = None,
          game_outcome: float = 0.0) -> GameHistory:
    """Synthetic GameHistory with N actions matching `rewards`. Default
    root_values are zero (so the bootstrap term contributes nothing), letting
    tests isolate the n-step reward sum."""
    g = GameHistory(game_name="chess")
    n = len(rewards)
    rv = root_values if root_values is not None else [0.0] * n
    assert len(rv) == n, "root_values must match rewards length"
    for t in range(n):
        g.observations.append(torch.zeros(3, 4, 4))
        g.actions.append(t % ACTION_SPACE)
        g.rewards.append(float(rewards[t]))
        g.policies.append(np.full(ACTION_SPACE, 1.0 / ACTION_SPACE, dtype=np.float32))
        g.root_values.append(float(rv[t]))
        g.legal_actions_list.append(list(range(ACTION_SPACE)))
    g.observations.append(torch.zeros(3, 4, 4))
    g.game_outcome = float(game_outcome)
    return g


def _value_at(g: GameHistory, state_index: int, td_steps: int,
              num_unroll_steps: int = 0, discount: float = 1.0) -> float:
    _, _, _, values, _, _ = g.make_target(
        state_index=state_index,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        discount=discount,
        action_space_size=ACTION_SPACE,
    )
    return float(values[0])


# --- Parity-flipped n-step sum -------------------------------------------------

def test_td_n_step_terminal_in_window_flips_parity_to_opposite_stm():
    """5-ply game where white mates on action j=4 (even, white's move).
    Sampling at state_index=1 (STM=black) with td_steps=4 means the 4-step
    sum reaches the terminal reward. From black's POV that win is a -1.

    Pre-fix (no parity flip): the loop added +1.0 (terminal r is +1 in white's
    POV but it gets summed as-is). Post-fix: the +1 reward at j=4 has
    opposite parity to idx=1, so it is added as -1.
    """
    g = _game([0.0, 0.0, 0.0, 0.0, 1.0])  # white wins at j=4
    v = _value_at(g, state_index=1, td_steps=4)
    assert v == pytest.approx(-1.0), (
        f"expected -1.0 (black's losing POV), got {v}; pre-fix would give +1.0"
    )


def test_td_n_step_same_parity_unchanged():
    """When the rewarded ply and the sample idx have the same parity, the
    n-step sum behaves identically pre/post fix. This is the degenerate
    case where the bug was invisible."""
    g = _game([0.0, 0.0, 0.0, 0.0, 1.0])
    # state_index=0 (white STM), terminal reward at j=4 (white STM) →
    # same parity → +1. Use td_steps=5 so the loop includes j=4 (the
    # range upper bound is exclusive on bootstrap_idx).
    v = _value_at(g, state_index=0, td_steps=5)
    assert v == pytest.approx(1.0)


def test_td_n_step_interior_reward_parity():
    """Interior non-zero reward at an opposite-parity ply gets sign-flipped.
    Construct rewards = [0, 0.5, 0, 0] so the only non-zero reward is at
    j=1 (black's move, +0.5 in black's POV). Sampling at state_index=0
    (white STM) with td_steps=2 must produce -0.5 (negated to white's POV)."""
    g = _game([0.0, 0.5, 0.0, 0.0])
    v = _value_at(g, state_index=0, td_steps=2, discount=1.0)
    assert v == pytest.approx(-0.5), (
        f"expected -0.5 (parity-flipped to white's POV), got {v}"
    )


def test_td_n_step_discount_applied_correctly():
    """The discount factor multiplies parity-adjusted rewards; both still
    survive the parity flip."""
    g = _game([0.0, 0.0, 1.0, 0.0])  # reward at j=2 (white)
    # idx=1 (black STM), td_steps=3 → bootstrap_idx=4 (out of range, skipped).
    # j=2 has opposite parity to idx=1 → multiplier = -1, discount = γ^(2-1) = γ.
    v = _value_at(g, state_index=1, td_steps=3, discount=0.9)
    assert v == pytest.approx(-0.9), (
        f"expected -0.9 (γ^1 · -1 · 1.0), got {v}"
    )


# --- Bootstrap term unchanged --------------------------------------------------

def test_td_n_step_bootstrap_parity_still_flips():
    """The bootstrap term's parity flip on root_values[bootstrap_idx] was
    already correct pre-fix and must stay correct post-fix."""
    # Zero rewards everywhere; only the bootstrap drives the value.
    g = _game([0.0] * 6, root_values=[0.0, 0.0, 0.0, 0.7, 0.0, 0.0])
    # idx=0 (white), bootstrap_idx=3 (black STM). Different parity → -1.
    v = _value_at(g, state_index=0, td_steps=3, discount=1.0)
    assert v == pytest.approx(-0.7), f"bootstrap parity flip broken; got {v}"

    # Same parity case.
    g2 = _game([0.0] * 6, root_values=[0.0, 0.0, 0.0, 0.0, 0.7, 0.0])
    v2 = _value_at(g2, state_index=0, td_steps=4, discount=1.0)
    assert v2 == pytest.approx(0.7), f"bootstrap parity match broken; got {v2}"


# --- MC path (td_steps=-1) unchanged ------------------------------------------

def test_td_steps_minus_one_uses_game_outcome_branch_unchanged():
    """The MC branch (td_steps=-1) is independent of the n-step sum and
    must still produce sign(idx) * game_outcome."""
    g = _game([0.0, 0.0, 0.0, 0.0, 1.0], game_outcome=1.0)
    # idx=0 (white) → +1; idx=1 (black) → -1.
    assert _value_at(g, state_index=0, td_steps=-1) == pytest.approx(1.0)
    assert _value_at(g, state_index=1, td_steps=-1) == pytest.approx(-1.0)
