"""Terminal-root policy target is a benign all-zeros no-op (bug_hunt_2026_06_13.md §E).

`ReplayBuffer.sample_batch` samples a root position with
`pos = np.random.randint(0, len(game))`, where `len(game) == len(observations)
== len(actions) + 1`. That range includes the TERMINAL observation index
(`idx == len(actions)`), for which there is no stored MCTS policy
(`self.policies` has one entry per non-terminal ply).

At that index `make_target` falls back to a ZEROS policy target. This is
INTENTIONAL and benign: a zeros target contributes exactly 0 cross-entropy
gradient (it pulls the policy head toward nothing), unlike a uniform/argmax
target which would inject a spurious gradient at a terminal state. The VALUE
target at the terminal index is still correctly computed (STM-POV outcome) and
trains normally.

These tests pin that invariant so a future refactor cannot silently turn the
benign zeros target into a harmful nonzero one, and cannot break the correct
terminal value target.
"""

import numpy as np
import pytest
import torch

from src.training.replay_buffer import GameHistory


ACTION_SPACE = 4


def _game(rewards: list[float], game_outcome: float = 0.0) -> GameHistory:
    """Synthetic GameHistory with N actions/policies/rewards and N+1 obs.

    Policies are non-uniform one-hots so that, if the terminal fallback were
    ever (wrongly) made to copy or interpolate a real policy, the test would
    see a nonzero/uneven target instead of the expected zeros.
    """
    g = GameHistory(game_name="chess")
    n = len(rewards)
    for t in range(n):
        g.observations.append(torch.zeros(3, 4, 4))
        g.actions.append(t % ACTION_SPACE)
        g.rewards.append(float(rewards[t]))
        p = np.zeros(ACTION_SPACE, dtype=np.float32)
        p[t % ACTION_SPACE] = 1.0  # distinct one-hot per ply
        g.policies.append(p)
        g.root_values.append(0.0)
        g.legal_actions_list.append(list(range(ACTION_SPACE)))
    # The extra terminal observation (no action/policy/reward for it).
    g.observations.append(torch.zeros(3, 4, 4))
    g.game_outcome = float(game_outcome)
    return g


def _target_at(g: GameHistory, state_index: int, td_steps: int = -1,
               num_unroll_steps: int = 0, discount: float = 1.0):
    return g.make_target(
        state_index=state_index,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        discount=discount,
        action_space_size=ACTION_SPACE,
    )


def test_terminal_index_is_in_root_sampling_range():
    """Document the precondition: randint(0, len(game)) can pick the terminal
    index, since len(game) == len(actions) + 1."""
    g = _game([0.0, 0.0, 0.0, 0.0, 1.0])
    terminal_idx = len(g.actions)               # == 5
    assert terminal_idx == len(g) - 1           # in [0, len(g)) → reachable
    assert terminal_idx >= len(g.policies)       # no stored policy there


def test_terminal_root_policy_target_is_all_zeros():
    """Sampling at the terminal index yields a ZEROS policy target → zero CE
    gradient. (A uniform target would be 0.25 per entry and would pull the
    policy head; a one-hot would be a spurious hard target. Neither must occur.)
    """
    g = _game([0.0, 0.0, 0.0, 0.0, 1.0])
    terminal_idx = len(g.actions)
    _, _, _, _, _, policies = _target_at(g, state_index=terminal_idx)
    policy0 = np.asarray(policies[0], dtype=np.float32)

    assert policy0.shape == (ACTION_SPACE,)
    assert np.count_nonzero(policy0) == 0, (
        f"terminal-root policy target must be all-zeros (benign no-op), "
        f"got {policy0!r}"
    )
    # Guard against the specific harmful regressions explicitly.
    uniform = np.full(ACTION_SPACE, 1.0 / ACTION_SPACE, dtype=np.float32)
    assert not np.allclose(policy0, uniform), "must not be a uniform target"
    assert policy0.sum() == pytest.approx(0.0), "must not be any normalized dist"


def test_terminal_root_value_target_is_correct_nonzero_stm_pov():
    """The value target at the terminal index is still correct: under the MC
    branch (td_steps=-1) it is sign(idx) * game_outcome from STM POV.

    With an odd number of actions (5), the terminal index is 5 (STM would be
    black-to-move-after-white's-5th-ply, i.e. odd parity). game_outcome=+1
    (white wins) → terminal value target = -1 from that POV. This is nonzero
    and correct, proving the position still trains its value head.
    """
    g = _game([0.0, 0.0, 0.0, 0.0, 1.0], game_outcome=1.0)
    terminal_idx = len(g.actions)               # 5 (odd)
    _, _, _, values, _, _ = _target_at(g, state_index=terminal_idx, td_steps=-1)
    sign = 1.0 if (terminal_idx % 2 == 0) else -1.0
    assert float(values[0]) == pytest.approx(sign * g.game_outcome)
    assert float(values[0]) != 0.0, "terminal value target must be nonzero here"


def test_non_terminal_root_policy_target_is_the_stored_policy():
    """Sanity contrast: a non-terminal index returns the real (nonzero) stored
    MCTS policy, so the zeros target is uniquely a terminal-index artifact and
    not a blanket suppression of policy gradients."""
    g = _game([0.0, 0.0, 0.0, 0.0, 1.0])
    _, _, _, _, _, policies = _target_at(g, state_index=2)
    policy0 = np.asarray(policies[0], dtype=np.float32)
    assert np.count_nonzero(policy0) > 0
    np.testing.assert_allclose(policy0, g.policies[2])
