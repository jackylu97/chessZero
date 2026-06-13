"""Tests for TensorMCTS (src/mcts/tensor_mcts.py).

Runs everything on CPU device — torch.multinomial, torch.distributions.Dirichlet,
and our gather/scatter ops all work on CPU, so these tests don't require CUDA.
A separate GPU bench/smoke lives outside the unit-test surface.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.mcts.mcts import BatchedMCTS, MCTSNode, select_action
from src.mcts.tensor_mcts import TensorMCTS


@dataclass
class StubGame:
    action_space_size: int = 9


@dataclass
class StubConfig:
    discount: float = 1.0
    num_simulations: int = 16
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    sample_k: int | None = None
    use_gumbel: bool = False


class StubNetwork(torch.nn.Module):
    """Minimal network for MCTS testing.

    Keeps small linear layers so .parameters() is non-empty (TensorMCTS reads
    parameter dtype for the recurrent_inference cast). Outputs are
    deterministic functions of the input batch shape.
    """

    def __init__(self, action_space: int = 9, hidden_planes: int = 4,
                 latent_h: int = 1, latent_w: int = 1):
        super().__init__()
        self.action_space = action_space
        self.hidden_planes = hidden_planes
        self.latent_h = latent_h
        self.latent_w = latent_w
        # A single learnable param so .parameters() yields something.
        self.dummy = torch.nn.Linear(1, 1)

    def initial_inference(self, obs: torch.Tensor):
        b = obs.shape[0]
        hidden = torch.zeros(b, self.hidden_planes, self.latent_h, self.latent_w)
        policy_logits = torch.zeros(b, self.action_space)
        value = torch.zeros(b, 1)
        return hidden, policy_logits, value

    def recurrent_inference(self, hidden: torch.Tensor, action: torch.Tensor):
        b = hidden.shape[0]
        next_hidden = torch.zeros(b, self.hidden_planes, self.latent_h, self.latent_w)
        reward = torch.zeros(b, 1)
        policy_logits = torch.zeros(b, self.action_space)
        value = torch.zeros(b, 1)
        return next_hidden, reward, policy_logits, value


class RewardingNetwork(StubNetwork):
    """Network that gives a +1 reward for one specific action (winning move).

    Used to verify that PUCT under TensorMCTS heads TOWARD the winning action.
    """

    def __init__(self, winning_action: int, action_space: int = 9, **kw):
        super().__init__(action_space=action_space, **kw)
        self.winning_action = winning_action

    def recurrent_inference(self, hidden: torch.Tensor, action: torch.Tensor):
        next_hidden, _, policy_logits, value = super().recurrent_inference(hidden, action)
        b = hidden.shape[0]
        # +1 reward when the winning action was taken.
        reward = (action == self.winning_action).float().view(b, 1)
        return next_hidden, reward, policy_logits, value


def _make_obs_batch(n: int) -> list[torch.Tensor]:
    """Stub observation tensors. TensorMCTS just torch.stack()'s them."""
    return [torch.zeros(3, 3, 3) for _ in range(n)]


def _make_legals(n: int, action_space: int = 9) -> list[list[int]]:
    return [list(range(action_space)) for _ in range(n)]


# --- Smoke ---------------------------------------------------------------


def test_run_batch_smoke_n4_sims8():
    """End-to-end run: no NaN, no exceptions, sensible output shapes."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=8, sample_k=5)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    roots = mcts.run_batch(_make_obs_batch(4), _make_legals(4), add_noise=True)

    assert len(roots) == 4
    for root in roots:
        assert isinstance(root, MCTSNode)
        # All visited slots account for one root visit per sim.
        # (Visits counted at root after each backprop.)
        assert root.visit_count == config.num_simulations
        # Sampled actions must be in legal range.
        assert (root.child_actions >= 0).all()
        assert (root.child_actions < 9).all()
        # Per-action visit counts sum to num_simulations.
        assert root.child_visits.sum() == pytest.approx(config.num_simulations)
        # Priors after dedup sum to ~1 (since β̂ over slots = 1/K, dedup sums
        # back to 1).
        assert root.child_priors.sum() == pytest.approx(1.0, abs=1e-5)
        # MCTSNode.value finite.
        assert np.isfinite(root.value)


def test_run_batch_no_dirichlet():
    """add_noise=False path runs without invoking the Dirichlet sampler."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=4, sample_k=5)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    roots = mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
    assert all(r.visit_count == 4 for r in roots)


# --- Backprop sign convention --------------------------------------------


def test_backprop_alternates_sign_with_zero_rewards():
    """Mover-POV: with reward=0 along path, root.value_sum sums to ±V_leaf
    alternating per ply. With V_leaf = 0 here (zero net), value_sum = 0
    everywhere — the meaningful test is "no NaN, root value finite".

    For a non-trivial sign check, see ``test_winning_action_drives_root_visits``
    which verifies search behavior under a non-zero reward setup.
    """
    torch.manual_seed(0)
    config = StubConfig(num_simulations=8, sample_k=3, discount=1.0)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    roots = mcts.run_batch(_make_obs_batch(1), _make_legals(1), add_noise=False)
    # All rewards / values are 0 → root value_sum = 0.
    assert roots[0].value == pytest.approx(0.0, abs=1e-6)


# --- Search behavior under a winning action ------------------------------


def test_winning_action_drives_root_visits():
    """When one action gives +1 reward, MCTS should concentrate visits on it.

    Mirrors the ``test_select_child_prefers_winning_child`` regression in
    test_mcts.py — the mover-POV sign convention must push selection TOWARD
    the winning child (positive Q under reward=+1), not away from it.
    """
    torch.manual_seed(0)
    winning = 4
    config = StubConfig(
        num_simulations=64, sample_k=9, discount=1.0, dirichlet_epsilon=0.0
    )
    mcts = TensorMCTS(
        RewardingNetwork(winning_action=winning, action_space=9),
        StubGame(action_space_size=9),
        config,
        device="cpu",
    )
    roots = mcts.run_batch(_make_obs_batch(1), _make_legals(1), add_noise=False)
    root = roots[0]

    # The winning action must be present among root children (uniform softmax
    # → all 9 legal actions sampled with high probability across K=9 draws).
    if winning in root.child_actions.tolist():
        winning_pos = int(np.where(root.child_actions == winning)[0][0])
        winning_visits = root.child_visits[winning_pos]
        other_max = root.child_visits[
            np.arange(len(root.child_visits)) != winning_pos
        ].max() if len(root.child_visits) > 1 else 0
        # Winning slot should receive a clear majority of visits.
        assert winning_visits > other_max, (
            f"winning action visits {winning_visits} not > other-max {other_max}"
        )
    else:
        # Sampling missed it (vanishingly unlikely with 9 draws over 9 actions
        # under uniform π — but possible with replacement). Re-seed fail-safe.
        pytest.skip("Winning action not sampled at root (replacement variance).")


# --- Visit-count accounting ----------------------------------------------


def test_root_visit_count_matches_num_simulations():
    """Every backprop pass increments root.visit_count by 1."""
    torch.manual_seed(0)
    for sims in [1, 4, 16, 32]:
        config = StubConfig(num_simulations=sims, sample_k=4)
        mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
        roots = mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
        for r in roots:
            assert r.visit_count == sims, f"sims={sims}, got {r.visit_count}"
            assert r.child_visits.sum() == pytest.approx(sims)


# --- Compat shim ---------------------------------------------------------


def test_compat_shim_dedupes_duplicate_slots():
    """Option (b) keeps duplicate sampled actions across slots; the compat
    shim must aggregate them into one entry per unique action so
    ``select_action`` sees the right policy distribution.
    """
    torch.manual_seed(42)
    # Sample K=20 over an action_space=4 — heavy duplication expected.
    config = StubConfig(num_simulations=8, sample_k=20)
    mcts = TensorMCTS(StubNetwork(action_space=4), StubGame(action_space_size=4),
                      config, device="cpu")
    roots = mcts.run_batch(
        [torch.zeros(3, 3, 3) for _ in range(1)],
        [list(range(4))],
        add_noise=False,
    )
    root = roots[0]
    # Unique actions only — dedup worked.
    assert len(root.child_actions) == len(set(root.child_actions.tolist()))
    # Priors sum to 1 after dedup (sum of 1/K across K slots = 1).
    assert root.child_priors.sum() == pytest.approx(1.0, abs=1e-5)


def test_select_action_works_on_returned_root():
    """The returned MCTSNode must be drop-in for the existing select_action."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=16, sample_k=5)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    roots = mcts.run_batch(_make_obs_batch(1), _make_legals(1), add_noise=True)
    action, probs = select_action(roots[0], temperature=1.0)
    assert 0 <= action < 9
    assert probs.shape[0] == int(roots[0].child_actions.max()) + 1
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


# --- Re-run / cache reuse -------------------------------------------------


def test_run_batch_reuses_allocated_storage():
    """Second run with same N must not re-allocate (correctness check, not perf)."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=8, sample_k=4)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    mcts.run_batch(_make_obs_batch(3), _make_legals(3), add_noise=False)
    n_id = id(mcts.node_visits)
    mcts.run_batch(_make_obs_batch(3), _make_legals(3), add_noise=False)
    assert id(mcts.node_visits) == n_id, "tree storage should be reused"


def test_run_batch_reallocates_when_n_changes():
    """Different N → re-allocation."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=4, sample_k=4)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
    n_id = id(mcts.node_visits)
    mcts.run_batch(_make_obs_batch(5), _make_legals(5), add_noise=False)
    assert id(mcts.node_visits) != n_id


# --- fp16 storage --------------------------------------------------------


def test_fp16_hidden_storage_runs_end_to_end():
    """Hidden states stored as fp16; cast back to compute dtype before each
    recurrent_inference call. Smoke check: no dtype mismatch errors and
    sensible output."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=8, sample_k=5)
    mcts = TensorMCTS(
        StubNetwork(), StubGame(), config, device="cpu",
        hidden_dtype=torch.float16,
    )
    roots = mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
    assert mcts.node_hidden.dtype == torch.float16
    for r in roots:
        assert r.visit_count == config.num_simulations
        assert np.isfinite(r.value)


# --- Soft parity vs. BatchedMCTS -----------------------------------------


def test_visit_distribution_concentrates_under_rewarding_network():
    """Both BatchedMCTS and TensorMCTS should concentrate visits on the
    winning action under a deterministic-reward network.

    Bit-equivalence is not achievable (CPU vs. GPU FP + different RNG paths
    in multinomial). We just verify both pick the same winner. Uses
    sample_k < action_space so both implementations exercise the Sampled
    MuZero path; sample_k chosen large enough that the winning action is
    sampled at root with overwhelming probability under uniform π
    (P(missed) = (1−1/A)^K = (19/20)^15 ≈ 0.46, so we re-seed if missed
    rather than asserting hard).
    """
    winning = 7
    action_space = 20
    config = StubConfig(
        num_simulations=256, sample_k=15, discount=1.0,
        dirichlet_epsilon=0.0,
    )
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(action_space))]

    # Hunt for a seed where both samplers happen to draw the winning action
    # at root — otherwise the assertion is meaningless. With ~50% miss-rate,
    # a few attempts always find one.
    found_seed = None
    for seed in range(20):
        np.random.seed(seed)
        torch.manual_seed(seed)
        net1 = RewardingNetwork(winning_action=winning, action_space=action_space)
        net2 = RewardingNetwork(winning_action=winning, action_space=action_space)
        bmcts = BatchedMCTS(
            net1, StubGame(action_space_size=action_space), config, device="cpu"
        )
        tmcts = TensorMCTS(
            net2, StubGame(action_space_size=action_space), config, device="cpu"
        )
        bmcts_root = bmcts.run_batch(obs, legals, add_noise=False)[0]
        tmcts_root = tmcts.run_batch(obs, legals, add_noise=False)[0]
        if (
            winning in bmcts_root.child_actions.tolist()
            and winning in tmcts_root.child_actions.tolist()
        ):
            found_seed = seed
            break
    assert found_seed is not None, "Couldn't find a seed where both samplers drew the winning action at root."

    bmcts_winner = int(bmcts_root.child_actions[int(np.argmax(bmcts_root.child_visits))])
    tmcts_winner = int(tmcts_root.child_actions[int(np.argmax(tmcts_root.child_visits))])
    assert bmcts_winner == winning, f"BatchedMCTS picked {bmcts_winner} (expected {winning})"
    assert tmcts_winner == winning, f"TensorMCTS picked {tmcts_winner} (expected {winning})"


# --- Multiple games run independently ------------------------------------


def test_multiple_games_run_independently():
    """When game A has winning action X and game B has winning action Y,
    each game's visits should concentrate on its own winning action.

    Uses K > action_space (heavy duplication) so both winning actions are
    almost certainly sampled at root despite uniform π.
    """
    action_space = 5
    win_a, win_b = 0, 4

    class TwoGameNetwork(StubNetwork):
        def __init__(self):
            super().__init__(action_space=action_space)

        def recurrent_inference(self, hidden, action):
            next_hidden, _, policy_logits, value = super().recurrent_inference(hidden, action)
            b = hidden.shape[0]
            # Reward the winning action ONLY at depth 1 (i.e. when expanded from
            # the ROOT, whose hidden state is all-zero). The root sets the
            # winning action's child reward to +1; we then emit a NON-zero
            # next_hidden so deeper expansions see a non-root parent and never
            # re-award. Without this guard the (degenerate) network would reward
            # the same action at every depth, and at discount=1.0 the opponent
            # collecting that same reward one ply deeper makes the "winning"
            # action no longer strictly dominant — a brittle expectation that
            # the old buggy duplicate-slot structure happened to satisfy but the
            # correct (deduped) search does not. Gating to depth 1 makes the
            # winning action unambiguously best, matching the test's intent.
            is_root_parent = (hidden.reshape(b, -1).abs().sum(dim=1) == 0)  # [b]
            reward = torch.zeros(b, 1)
            if is_root_parent[0]:
                reward[0, 0] = float(action[0].item() == win_a)
            if b > 1 and is_root_parent[1]:
                reward[1, 0] = float(action[1].item() == win_b)
            # Non-zero hidden marks "below root" for the next recurrence.
            next_hidden = next_hidden + 1.0
            return next_hidden, reward, policy_logits, value

    # K=20 over action_space=5: per-action P(missed) = (4/5)^20 ≈ 0.012, but
    # we need TWO specific actions to land across two games drawn from the
    # same RNG state, so combined miss-rate is closer to ~5%/seed in practice.
    # Sweep enough seeds to make a skip vanishingly unlikely.
    config = StubConfig(
        num_simulations=128, sample_k=20, discount=1.0,
        dirichlet_epsilon=0.0,
    )
    for seed in range(50):
        torch.manual_seed(seed)
        mcts = TensorMCTS(TwoGameNetwork(), StubGame(action_space_size=action_space),
                          config, device="cpu")
        roots = mcts.run_batch(
            [torch.zeros(3, 3, 3), torch.zeros(3, 3, 3)],
            [list(range(action_space)), list(range(action_space))],
            add_noise=False,
        )
        if (
            win_a in roots[0].child_actions.tolist()
            and win_b in roots[1].child_actions.tolist()
        ):
            break
    else:
        pytest.skip("Couldn't sample both winning actions at root within 50 seeds.")

    a_winner = int(roots[0].child_actions[int(np.argmax(roots[0].child_visits))])
    b_winner = int(roots[1].child_actions[int(np.argmax(roots[1].child_visits))])
    assert a_winner == win_a, f"game 0 picked {a_winner}, expected {win_a}"
    assert b_winner == win_b, f"game 1 picked {b_winner}, expected {win_b}"


# --- Pin: PUCT formula ---------------------------------------------------


def test_select_picks_slot_with_highest_puct_score():
    """Hand-construct a tree state, hand-compute PUCT scores, verify _select
    picks the argmax slot.

    Pins the formula:
        pb_c = log((N + 19652 + 1) / 19652) + 1.25
        prior_score = pb_c * prior * sqrt(N) / (1 + visits)
        child_value = value_sum / max(1, visits)   if visits > 0 else 0
        raw_q = reward - discount * child_value
        normalized_q = (raw_q - mm_min) / (mm_max - mm_min)   if has_range else raw_q
        value_score = normalized_q   if visits > 0 else 0
        score = prior_score + value_score
    """
    config = StubConfig(num_simulations=1, sample_k=4, discount=1.0,
                        dirichlet_epsilon=0.0)
    mcts = TensorMCTS(StubNetwork(), StubGame(action_space_size=9),
                      config, device="cpu")
    mcts._allocate(n=1, num_simulations=1, hidden_shape=(4, 1, 1))
    mcts._reset()

    mcts.node_count.fill_(1)
    mcts.node_visits[0, 0] = 10
    actions = [0, 1, 2, 3]
    priors = [0.4, 0.3, 0.2, 0.1]
    visits = [3, 2, 1, 4]
    value_sums = [0.6, 0.0, 0.5, 0.8]
    rewards = [0.0, 0.0, 0.0, 0.0]
    mcts.child_actions[0, 0, :] = torch.tensor(actions, dtype=torch.int32)
    mcts.child_priors[0, 0, :] = torch.tensor(priors)
    mcts.child_visits[0, 0, :] = torch.tensor(visits, dtype=torch.int32)
    mcts.child_value_sum[0, 0, :] = torch.tensor(value_sums)
    mcts.child_rewards[0, 0, :] = torch.tensor(rewards)
    mcts.mm_min[0] = -1.0
    mcts.mm_max[0] = +1.0

    _, _, _, leaf_slot, _, _ = mcts._select()

    pb_c = math.log((10 + 19652 + 1) / 19652) + 1.25
    sqrt_n = math.sqrt(10)
    expected = []
    for i in range(4):
        prior_score = pb_c * priors[i] * sqrt_n / (1.0 + visits[i])
        child_value = value_sums[i] / max(1.0, visits[i])
        raw_q = rewards[i] - 1.0 * child_value
        norm_q = (raw_q - (-1.0)) / 2.0
        value_score = norm_q if visits[i] > 0 else 0.0
        expected.append(prior_score + value_score)
    expected_winner = int(np.argmax(expected))

    assert leaf_slot[0].item() == expected_winner, (
        f"_select chose slot {leaf_slot[0].item()}, expected {expected_winner}; "
        f"hand-computed scores: {expected}"
    )


# --- Pin: MinMaxStats establishment boundary -----------------------------


def _build_mm_test_tree():
    """Construct a 2-slot tree where raw_q vs normalized_q give different argmax.

    Slot 0: high prior (0.9), small negative raw_q (-0.1) → prior_score ≫ |value_score|.
    Slot 1: low prior (0.1), small positive raw_q (+0.1) → small prior_score.

    Without normalization (raw_q in [-0.1, +0.1] range): prior_score dominates,
    slot 0 wins.

    With normalization over mm_range [-0.1, +0.1] (span=0.2): raw_q snaps to
    {0, 1}, and value_score=1 dominates the prior_score gap → slot 1 wins.
    """
    config = StubConfig(num_simulations=1, sample_k=2, discount=1.0,
                        dirichlet_epsilon=0.0)
    mcts = TensorMCTS(StubNetwork(), StubGame(action_space_size=4),
                      config, device="cpu")
    mcts._allocate(n=1, num_simulations=1, hidden_shape=(4, 1, 1))
    mcts._reset()
    mcts.node_count.fill_(1)
    mcts.node_visits[0, 0] = 10
    mcts.child_actions[0, 0, :] = torch.tensor([0, 1], dtype=torch.int32)
    mcts.child_priors[0, 0, :] = torch.tensor([0.9, 0.1])
    mcts.child_visits[0, 0, :] = torch.tensor([5, 5], dtype=torch.int32)
    mcts.child_value_sum[0, 0, :] = torch.tensor([+0.5, -0.5])
    mcts.child_rewards[0, 0, :] = torch.zeros(2)
    return mcts


def test_select_uses_raw_q_when_mm_range_empty():
    """has_range=False (mm sentinels at +inf/-inf) → value_score = raw_q.

    Slot 0's high prior_score wins by 0.527 vs slot 1's tiny ~0.166. Adding
    the small raw_q value scores (-0.1 vs +0.1) doesn't flip the ranking, so
    slot 0 still wins.
    """
    mcts = _build_mm_test_tree()  # mm_min=+inf, mm_max=-inf from _reset
    _, _, _, leaf_slot, _, _ = mcts._select()
    assert leaf_slot[0].item() == 0, "without mm normalization, prior_score dominates"


def test_select_uses_normalized_q_when_mm_range_nonempty():
    """has_range=True → value_score = (raw_q - mm_min) / (mm_max - mm_min).

    Same tree as the raw_q test, but mm_range=[-0.1, +0.1] (span=0.2). raw_q
    of -0.1 / +0.1 maps to normalized_q of 0 / 1 — value_score for slot 1
    becomes 1.0, dwarfing the prior_score gap → slot 1 wins.
    """
    mcts = _build_mm_test_tree()
    mcts.mm_min[0] = -0.1
    mcts.mm_max[0] = +0.1
    _, _, _, leaf_slot, _, _ = mcts._select()
    assert leaf_slot[0].item() == 1, "with mm normalization, value_score (=1.0) flips winner"


# --- Pin: backprop multi-ply sign flip + child_rewards mirror ------------


@pytest.mark.parametrize(
    "leaf_v,reward_root,reward_mid,reward_leaf,discount,exp_root,exp_mid,exp_leaf",
    [
        # zero rewards, γ=1: classic alternating ±V_leaf flip per ply.
        (1.0, 0.0, 0.0, 0.0, 1.0, +1.0, -1.0, +1.0),
        # reward at mid-node: stacks into the root accumulation.
        # d=2: vs[leaf]=+1, value = 0 - 1·1 = -1
        # d=1: vs[mid]=-1, value = 0.5 - 1·(-1) = +1.5
        # d=0: vs[root]=+1.5
        (1.0, 0.0, 0.5, 0.0, 1.0, +1.5, -1.0, +1.0),
        # discount=0.9 — value attenuates per ply.
        # d=2: vs[leaf]=+1, value = 0 - 0.9·1 = -0.9
        # d=1: vs[mid]=-0.9, value = 0 - 0.9·(-0.9) = +0.81
        # d=0: vs[root]=+0.81
        (1.0, 0.0, 0.0, 0.0, 0.9, +0.81, -0.9, +1.0),
    ],
    ids=["zero_rewards_g1", "intermediate_reward_g1", "zero_rewards_g0.9"],
)
def test_backprop_path_accumulates_with_sign_flip(
    leaf_v, reward_root, reward_mid, reward_leaf, discount,
    exp_root, exp_mid, exp_leaf,
):
    """3-ply backprop (root→mid→leaf) writes the right value at each node.

    Mover-POV: ``value`` at depth d is in node-d's frame. Per-ply update is:
    1. node_value_sum[d] += value
    2. value_to_propagate_up = node.reward[d] - γ · value
    """
    config = StubConfig(num_simulations=4, sample_k=2, discount=discount)
    mcts = TensorMCTS(StubNetwork(), StubGame(action_space_size=4),
                      config, device="cpu")
    mcts._allocate(n=1, num_simulations=4, hidden_shape=(4, 1, 1))
    mcts._reset()

    # Pre-existing tree: root(0) → mid(1) at root.slot=0. Leaf is the new node
    # being added during this backprop call. node_count=2 because root and mid
    # are already allocated; _backprop receives new_node_idx=2 from the
    # caller (normally _expand sets node_count[new_l] before _backprop runs).
    mcts.node_count.fill_(3)
    mcts.node_reward[0, 0] = reward_root
    mcts.node_reward[0, 1] = reward_mid
    mcts.node_reward[0, 2] = reward_leaf
    mcts.parent_idx[0, 1] = 0
    mcts.parent_child_slot[0, 1] = 0
    mcts.parent_idx[0, 2] = 1
    mcts.parent_child_slot[0, 2] = 0

    m = 5
    path_node_idx = torch.full((1, m), -1, dtype=torch.int32)
    path_node_idx[0, 0] = 0  # root at depth 0
    path_node_idx[0, 1] = 1  # mid at depth 1
    path_slot = torch.full((1, m), -1, dtype=torch.int32)
    path_slot[0, 1] = 0  # slot in root pointing to mid
    depth = torch.tensor([1], dtype=torch.int32)
    new_node_idx = torch.tensor([2], dtype=torch.int32)
    leaf_slot = torch.tensor([0], dtype=torch.int32)
    leaf_value = torch.tensor([leaf_v], dtype=torch.float32)

    mcts._backprop(path_node_idx, path_slot, depth, new_node_idx, leaf_slot, leaf_value)

    # Per-node accumulated value sums.
    assert mcts.node_value_sum[0, 0].item() == pytest.approx(exp_root)
    assert mcts.node_value_sum[0, 1].item() == pytest.approx(exp_mid)
    assert mcts.node_value_sum[0, 2].item() == pytest.approx(exp_leaf)

    # Each visited node gets visit_count += 1.
    assert mcts.node_visits[0, 0].item() == 1
    assert mcts.node_visits[0, 1].item() == 1
    assert mcts.node_visits[0, 2].item() == 1

    # Mirror to parent's child arrays: child stats track underlying-node stats.
    # child[root, slot=0] should mirror mid's running totals.
    assert mcts.child_visits[0, 0, 0].item() == 1
    assert mcts.child_value_sum[0, 0, 0].item() == pytest.approx(exp_mid)
    # child[mid, slot=0] should mirror leaf's running totals.
    assert mcts.child_visits[0, 1, 0].item() == 1
    assert mcts.child_value_sum[0, 1, 0].item() == pytest.approx(exp_leaf)


# --- Pin: child_rewards mirror in _expand --------------------------------


def test_expand_writes_leaf_reward_to_parent_child_rewards():
    """``_expand`` must mirror the new leaf's reward into
    ``child_rewards[parent, leaf_slot]`` so PUCT's raw_q picks it up on the
    next selection pass.
    """
    config = StubConfig(num_simulations=2, sample_k=2)
    mcts = TensorMCTS(
        RewardingNetwork(winning_action=2, action_space=4),
        StubGame(action_space_size=4),
        config, device="cpu",
    )
    mcts._allocate(n=1, num_simulations=2, hidden_shape=(4, 1, 1))
    mcts._reset()
    mcts.node_count.fill_(1)
    # Two slots at root: slot 0 → action 2 (winning, reward +1);
    # slot 1 → action 0 (non-winning, reward 0).
    mcts.child_actions[0, 0, 0] = 2
    mcts.child_actions[0, 0, 1] = 0

    leaf_parent = torch.tensor([0], dtype=torch.int32)  # root
    leaf_slot = torch.tensor([0], dtype=torch.int32)    # slot 0
    leaf_action = torch.tensor([2], dtype=torch.int32)  # winning action

    new_node_idx, _ = mcts._expand(leaf_parent, leaf_slot, leaf_action)

    # Mirrored into parent's child_rewards at the chosen slot.
    assert mcts.child_rewards[0, 0, 0].item() == pytest.approx(1.0)
    # Other slots untouched.
    assert mcts.child_rewards[0, 0, 1].item() == 0.0
    # Same reward also recorded on the new node.
    new_idx = int(new_node_idx[0].item())
    assert mcts.node_reward[0, new_idx].item() == pytest.approx(1.0)
    # Parent linkage and child_node_idx wiring.
    assert mcts.parent_idx[0, new_idx].item() == 0
    assert mcts.parent_child_slot[0, new_idx].item() == 0
    assert mcts.child_node_idx[0, 0, 0].item() == new_idx


# --- Reallocation triggers ------------------------------------------------


def test_run_batch_reallocates_when_num_simulations_changes():
    """Different M (num_simulations + 1) → fresh storage."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=4, sample_k=4)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
    n_id = id(mcts.node_visits)
    config.num_simulations = 8
    mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
    assert id(mcts.node_visits) != n_id, "tree storage should reallocate when M changes"
    # New M = 9 (sims 8 + root).
    assert mcts.node_visits.shape == (2, 9)


def test_run_batch_reallocates_when_hidden_shape_changes():
    """Different network hidden shape → fresh storage."""
    torch.manual_seed(0)
    config = StubConfig(num_simulations=4, sample_k=4)
    mcts = TensorMCTS(
        StubNetwork(hidden_planes=4, latent_h=1, latent_w=1),
        StubGame(), config, device="cpu",
    )
    mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
    h_id = id(mcts.node_hidden)
    # Swap the network for one with different hidden output shape.
    mcts.network = StubNetwork(hidden_planes=8, latent_h=2, latent_w=2)
    mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=False)
    assert id(mcts.node_hidden) != h_id
    # New shape: [N=2, M=5, C=8, H=2, W=2].
    assert mcts.node_hidden.shape == (2, 5, 8, 2, 2)


# --- Tree reuse correctness ----------------------------------------------


def test_run_batch_second_call_consistent_with_matched_seed():
    """Reusing the same allocated storage between calls must not leak state.

    With matched torch RNG seeds, two consecutive ``run_batch`` calls on the
    same instance should produce byte-identical roots — the second call
    relies entirely on ``_reset`` to clear leftovers from the first.
    """
    config = StubConfig(num_simulations=8, sample_k=4)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")

    torch.manual_seed(0)
    out1 = mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=True)

    torch.manual_seed(0)
    out2 = mcts.run_batch(_make_obs_batch(2), _make_legals(2), add_noise=True)

    for r1, r2 in zip(out1, out2):
        assert r1.visit_count == r2.visit_count
        assert r1.value_sum == pytest.approx(r2.value_sum)
        assert np.array_equal(r1.child_actions, r2.child_actions)
        assert np.allclose(r1.child_visits, r2.child_visits)
        assert np.allclose(r1.child_priors, r2.child_priors)
        assert np.allclose(r1.child_value_sums, r2.child_value_sums)
        assert np.allclose(r1.child_rewards, r2.child_rewards)


# --- Dirichlet noise quantitative mixing ---------------------------------


def test_dirichlet_noise_replaces_priors_at_eps_one():
    """``priors = (1−ε)·1/K + ε·noise``. At ε=1 priors equal the noise sample.

    Verify (a) the slot-priors still sum to 1 after replacement, (b) they
    are non-uniform (Dirichlet(0.3) concentrates mass on a few slots — the
    constant-1/K initial priors would yield exact uniform).
    """
    K = 4
    config = StubConfig(num_simulations=1, sample_k=K,
                        dirichlet_alpha=0.3, dirichlet_epsilon=1.0)
    torch.manual_seed(42)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    mcts.run_batch(_make_obs_batch(1), _make_legals(1), add_noise=True)

    priors = mcts.child_priors[0, 0, :]
    assert priors.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (priors >= 0).all()
    # Dirichlet(0.3) is highly skewed → not uniform 1/K.
    assert not torch.allclose(priors, torch.full((K,), 1.0 / K), atol=1e-3)


def test_dirichlet_noise_mixing_at_eps_half():
    """At ε=0.5 the blend is ``priors[a] = 0.5·β̂(a) + 0.5·noise[a]`` over the
    U UNIQUE sampled actions (Sampled-MuZero dedup), with the K−U padding slots
    held at prior 0.

    β̂(a) = count(a)/K (≥ 1/K since each unique action has count ≥ 1) and the
    Dirichlet noise is drawn over the U unique actions (Σ noise = 1, noise ∈
    [0, 1]). So for each VALID slot:
        Σ_valid priors = 0.5·1 + 0.5·1 = 1
        priors[a] ∈ [0.5·β̂(a), 0.5·β̂(a) + 0.5] ⊆ [0.5/K, 1]

    Padding slots stay exactly 0. A pure-replacement bug (noise overwrites the
    β̂ prior) would drop a valid slot below 0.5·β̂(a); a pure-addition bug
    (priors = β̂ + noise) would push Σ above 1.
    """
    K = 4
    config = StubConfig(num_simulations=1, sample_k=K,
                        dirichlet_alpha=0.3, dirichlet_epsilon=0.5)
    torch.manual_seed(42)
    mcts = TensorMCTS(StubNetwork(), StubGame(), config, device="cpu")
    mcts.run_batch(_make_obs_batch(1), _make_legals(1), add_noise=True)

    priors = mcts.child_priors[0, 0, :]
    actions = mcts.child_actions[0, 0, :]
    valid = actions != -1
    # Valid (unique-action) priors sum to 1; padding slots are exactly 0.
    assert priors[valid].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (priors[~valid] == 0).all(), \
        f"padding slots must be 0, got {priors[~valid].tolist()}"

    # Per valid slot: 0.5·β̂(a) ≤ priors[a] ≤ 0.5·β̂(a)+0.5. Lower-bound the
    # base by the minimum possible β̂ (= 1/K) and upper-bound by 1.
    lo = 0.5 / K - 1e-6
    hi = 1.0 + 1e-6
    vp = priors[valid]
    assert (vp >= lo).all(), f"valid priors {vp.tolist()} below lower bound {lo}"
    assert (vp <= hi).all(), f"valid priors {vp.tolist()} above upper bound {hi}"
