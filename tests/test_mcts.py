"""Tests for MCTS — covers both the loop and vectorized _select_child paths.

The loop and vectorized paths must produce identical results; the dispatched
entry point picks between them based on child count (_VECTORIZE_THRESHOLD).
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.mcts.mcts import (
    MCTS,
    BatchedMCTS,
    MCTSNode,
    MinMaxStats,
    select_action,
)


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


class StubNetwork:
    """Deterministic stub network — returns fixed-shape outputs for MCTS testing."""

    def __init__(self, action_space: int = 9, hidden_dim: int = 4):
        self.action_space = action_space
        self.hidden_dim = hidden_dim

    def initial_inference(self, obs: torch.Tensor):
        b = obs.shape[0]
        hidden = torch.zeros(b, self.hidden_dim)
        policy_logits = torch.zeros(b, self.action_space)
        value = torch.zeros(b, 1)
        return hidden, policy_logits, value

    def recurrent_inference(self, hidden: torch.Tensor, action: torch.Tensor):
        b = hidden.shape[0]
        next_hidden = torch.zeros(b, self.hidden_dim)
        reward = torch.zeros(b, 1)
        policy_logits = torch.zeros(b, self.action_space)
        value = torch.zeros(b, 1)
        return next_hidden, reward, policy_logits, value


def _make_node(k: int, seed: int = 42, parent_visits: int = 100) -> MCTSNode:
    """Build a fully-populated parent node with K children for selection tests."""
    rng = np.random.RandomState(seed)
    node = MCTSNode(visit_count=parent_visits)
    priors = rng.dirichlet(np.ones(k) * 0.3).astype(np.float64)
    visits = rng.randint(0, 20, size=k).astype(np.float64)
    rewards = rng.uniform(-1.0, 1.0, size=k).astype(np.float64)
    value_sums = rng.uniform(-5.0, 5.0, size=k).astype(np.float64)

    node.child_actions = np.arange(k, dtype=np.int64)
    node.child_priors = priors
    node.child_visits = visits.copy()
    node.child_rewards = rewards.copy()
    node.child_value_sums = value_sums.copy()
    node.children = []
    for i in range(k):
        c = MCTSNode(
            prior=float(priors[i]),
            visit_count=int(visits[i]),
            value_sum=float(value_sums[i]),
            reward=float(rewards[i]),
        )
        node.children.append(c)
    return node


def _make_mcts(action_space: int = 9) -> MCTS:
    return MCTS(
        network=StubNetwork(action_space),
        game=StubGame(action_space),
        config=StubConfig(),
        device="cpu",
    )


# --- _select_child parity tests ---------------------------------------------

@pytest.mark.parametrize("k", [2, 5, 9, 15, 25, 40, 60, 100, 500])
def test_loop_and_vectorized_return_same_action(k):
    """The two paths must agree on the chosen action for any K."""
    mcts = _make_mcts()
    for seed in range(5):
        node = _make_node(k, seed=seed)
        stats = MinMaxStats()
        stats.update(-0.5)
        stats.update(0.5)
        loop_out = mcts._select_child_loop(node, stats)
        vec_out = mcts._select_child_vectorized(node, stats)
        assert loop_out[0] == vec_out[0], f"action mismatch at k={k}, seed={seed}"
        assert loop_out[1] == vec_out[1], f"index mismatch at k={k}, seed={seed}"
        assert loop_out[2] is vec_out[2], f"child ref mismatch at k={k}, seed={seed}"


@pytest.mark.parametrize("k", [3, 9, 30, 100])
def test_select_child_parity_with_empty_min_max_stats(k):
    """When MinMaxStats has no range, value_score falls through differently — verify parity."""
    mcts = _make_mcts()
    node = _make_node(k, seed=7)
    stats = MinMaxStats()  # untouched — maximum < minimum, no range
    loop_out = mcts._select_child_loop(node, stats)
    vec_out = mcts._select_child_vectorized(node, stats)
    assert loop_out[:2] == vec_out[:2]


def test_select_child_parity_with_zero_visit_children():
    """Fresh children (visit_count=0) should score purely by prior."""
    mcts = _make_mcts()
    node = MCTSNode(visit_count=1)
    k = 9
    priors = np.array([0.05, 0.1, 0.4, 0.05, 0.1, 0.05, 0.1, 0.1, 0.05])
    node.child_actions = np.arange(k, dtype=np.int64)
    node.child_priors = priors
    node.child_visits = np.zeros(k)
    node.child_rewards = np.zeros(k)
    node.child_value_sums = np.zeros(k)
    node.children = [MCTSNode(prior=float(priors[i])) for i in range(k)]
    stats = MinMaxStats()

    loop_out = mcts._select_child_loop(node, stats)
    vec_out = mcts._select_child_vectorized(node, stats)
    assert loop_out[:2] == vec_out[:2]
    # Highest prior (index 2) should win when all visits are zero.
    assert loop_out[1] == 2


# --- Dispatch tests ---------------------------------------------------------

def test_dispatch_below_threshold_uses_loop(monkeypatch):
    mcts = _make_mcts()
    # K = threshold - 1 should hit the loop path.
    k = mcts._VECTORIZE_THRESHOLD - 1
    node = _make_node(k, seed=3)
    stats = MinMaxStats(); stats.update(-1.0); stats.update(1.0)

    calls = {"loop": 0, "vec": 0}
    real_loop = mcts._select_child_loop
    real_vec = mcts._select_child_vectorized
    monkeypatch.setattr(mcts, "_select_child_loop",
                        lambda n, s: (calls.__setitem__("loop", calls["loop"] + 1) or real_loop(n, s)))
    monkeypatch.setattr(mcts, "_select_child_vectorized",
                        lambda n, s: (calls.__setitem__("vec", calls["vec"] + 1) or real_vec(n, s)))
    mcts._select_child(node, stats)
    assert calls == {"loop": 1, "vec": 0}


def test_dispatch_at_threshold_uses_vectorized(monkeypatch):
    mcts = _make_mcts()
    k = mcts._VECTORIZE_THRESHOLD
    node = _make_node(k, seed=3)
    stats = MinMaxStats(); stats.update(-1.0); stats.update(1.0)

    calls = {"loop": 0, "vec": 0}
    real_loop = mcts._select_child_loop
    real_vec = mcts._select_child_vectorized
    monkeypatch.setattr(mcts, "_select_child_loop",
                        lambda n, s: (calls.__setitem__("loop", calls["loop"] + 1) or real_loop(n, s)))
    monkeypatch.setattr(mcts, "_select_child_vectorized",
                        lambda n, s: (calls.__setitem__("vec", calls["vec"] + 1) or real_vec(n, s)))
    mcts._select_child(node, stats)
    assert calls == {"loop": 0, "vec": 1}


# --- _expand tests ----------------------------------------------------------

def test_expand_populates_parallel_arrays():
    mcts = _make_mcts()
    node = MCTSNode()
    logits = torch.tensor([0.0, 1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0])
    legal = [1, 2, 3]
    mcts._expand(node, logits, legal)

    assert node.child_actions.tolist() == legal
    assert node.child_priors.shape == (3,)
    assert np.isclose(node.child_priors.sum(), 1.0)
    assert node.child_visits.tolist() == [0.0, 0.0, 0.0]
    assert node.child_rewards.tolist() == [0.0, 0.0, 0.0]
    assert node.child_value_sums.tolist() == [0.0, 0.0, 0.0]
    # Children are lazily allocated: slots start as None and get materialized on first visit.
    assert len(node.children) == 3
    assert all(c is None for c in node.children)


def test_expand_from_priors_populates_arrays():
    """_expand_from_priors is the pure-numpy entry used by BatchedMCTS —
    no softmax, no CPU-GPU transfer; caller supplies priors directly."""
    mcts = _make_mcts()
    node = MCTSNode()
    actions = np.array([3, 5, 7], dtype=np.int64)
    priors = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    mcts._expand_from_priors(node, actions, priors)

    assert np.array_equal(node.child_actions, actions)
    assert np.array_equal(node.child_priors, priors)
    assert node.child_visits.tolist() == [0.0, 0.0, 0.0]
    # Lazy allocation: slots are None until first traversal.
    assert len(node.children) == 3
    assert all(c is None for c in node.children)


# --- _add_dirichlet_noise ---------------------------------------------------

def test_dirichlet_noise_updates_child_priors_array():
    mcts = _make_mcts()
    node = MCTSNode(visit_count=0)
    k = 5
    base_priors = np.array([0.2] * k)
    node.child_actions = np.arange(k, dtype=np.int64)
    node.child_priors = base_priors.copy()
    node.child_visits = np.zeros(k)
    node.child_rewards = np.zeros(k)
    node.child_value_sums = np.zeros(k)
    # Mix of pre-materialized and lazy children — both paths must stay consistent.
    node.children = [MCTSNode(prior=float(base_priors[i])) if i < 2 else None
                     for i in range(k)]

    np.random.seed(0)
    mcts._add_dirichlet_noise(node)

    # Priors still sum to ~1 and noise actually applied.
    assert node.child_priors.sum() == pytest.approx(1.0)
    assert not np.allclose(node.child_priors, 0.2)
    # Pre-materialized children had their .prior synced; lazy slots stay None
    # and will pick up the noised prior when materialized via _get_or_create_child.
    for i, child in enumerate(node.children):
        if child is not None:
            assert child.prior == pytest.approx(float(node.child_priors[i]))

    # A subsequent materialization reads the noised prior from the array.
    materialized = mcts._get_or_create_child(node, 3)
    assert materialized.prior == pytest.approx(float(node.child_priors[3]))


# --- _backpropagate ---------------------------------------------------------

def test_backpropagate_syncs_parent_arrays_with_children():
    """After backprop, child.visit_count/value_sum/reward must match parent arrays."""
    mcts = _make_mcts()
    # Build a 3-deep path: root → c1 → c2.
    root = _make_node(5, seed=1, parent_visits=0)
    c1 = root.children[2]
    # Expand c1.
    c1.child_actions = np.array([0, 1], dtype=np.int64)
    c1.child_priors = np.array([0.5, 0.5])
    c1.child_visits = np.zeros(2)
    c1.child_rewards = np.zeros(2)
    c1.child_value_sums = np.zeros(2)
    c1.children = [MCTSNode(prior=0.5), MCTSNode(prior=0.5)]
    c2 = c1.children[1]
    c2.reward = 0.3

    search_path = [root, c1, c2]
    path_indices = [-1, 2, 1]

    c1_prev_visits = c1.visit_count
    c2_prev_visits = c2.visit_count
    stats = MinMaxStats()
    mcts._backpropagate(search_path, path_indices, value=0.7, min_max_stats=stats)

    # Visits incremented.
    assert c2.visit_count == c2_prev_visits + 1
    assert c1.visit_count == c1_prev_visits + 1
    assert root.visit_count == 1  # root started at 0

    # Parent arrays sync with child objects.
    assert root.child_visits[2] == c1.visit_count
    assert root.child_value_sums[2] == c1.value_sum
    assert root.child_rewards[2] == c1.reward
    assert c1.child_visits[1] == c2.visit_count
    assert c1.child_value_sums[1] == c2.value_sum
    assert c1.child_rewards[1] == c2.reward


def test_backpropagate_alternates_value_sign():
    """Signs alternate along the path (two-player zero-sum)."""
    mcts = _make_mcts()
    root = _make_node(3, seed=5, parent_visits=0)
    # Reset children's stats so we can observe deltas cleanly.
    for c in root.children:
        c.visit_count = 0
        c.value_sum = 0.0
        c.reward = 0.0
    root.child_visits[:] = 0
    root.child_value_sums[:] = 0
    root.child_rewards[:] = 0

    c = root.children[0]
    search_path = [root, c]
    path_indices = [-1, 0]

    mcts._backpropagate(search_path, path_indices, value=1.0, min_max_stats=MinMaxStats())
    # Leaf (depth=n-1): sign=+1 → +1.0. Root (depth=0): sign=-1 → -1.0.
    assert c.value_sum == pytest.approx(1.0)
    assert root.value_sum == pytest.approx(-1.0)


# --- MCTSNode helpers -------------------------------------------------------

def test_node_value_zero_visits_returns_zero():
    assert MCTSNode().value == 0.0


def test_node_value_nonzero_visits():
    node = MCTSNode(visit_count=4, value_sum=2.0)
    assert node.value == 0.5


def test_node_expanded_flag():
    node = MCTSNode()
    assert not node.expanded()
    node.children = [MCTSNode()]
    assert node.expanded()


# --- MinMaxStats ------------------------------------------------------------

def test_minmax_normalize_no_range_passthrough():
    stats = MinMaxStats()
    assert stats.normalize(0.5) == 0.5


def test_minmax_normalize_with_range():
    stats = MinMaxStats()
    stats.update(-1.0)
    stats.update(3.0)
    assert stats.normalize(1.0) == pytest.approx(0.5)
    assert stats.normalize(-1.0) == 0.0
    assert stats.normalize(3.0) == 1.0


# --- select_action ---------------------------------------------------------

def test_select_action_temperature_zero_picks_max_visited():
    root = MCTSNode()
    root.child_actions = np.array([2, 5, 7])
    root.child_visits = np.array([3.0, 10.0, 1.0])
    action, probs = select_action(root, temperature=0)
    assert action == 5
    assert probs[5] == 1.0
    assert sum(probs) == pytest.approx(1.0)


def test_select_action_temperature_one_distribution_sums_to_one():
    root = MCTSNode()
    root.child_actions = np.array([0, 1, 2])
    root.child_visits = np.array([1.0, 2.0, 3.0])
    np.random.seed(0)
    action, probs = select_action(root, temperature=1.0)
    assert action in [0, 1, 2]
    assert sum(probs) == pytest.approx(1.0)
    # Visit-count-proportional: probs == [1/6, 2/6, 3/6].
    assert probs[0] == pytest.approx(1 / 6)
    assert probs[1] == pytest.approx(2 / 6)
    assert probs[2] == pytest.approx(3 / 6)


# --- Integration: full MCTS.run --------------------------------------------

def test_mcts_run_produces_valid_tree():
    mcts = _make_mcts(action_space=9)
    legal = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    obs = torch.zeros(3, 3, 3)
    root = mcts.run(obs, legal, add_noise=False)

    assert root.visit_count == mcts.config.num_simulations
    assert len(root.children) == len(legal)
    assert int(root.child_visits.sum()) == mcts.config.num_simulations
    # Root arrays and child objects stay in sync throughout run().
    for i, child in enumerate(root.children):
        assert root.child_visits[i] == child.visit_count
        assert root.child_value_sums[i] == pytest.approx(child.value_sum)


def test_mcts_run_with_dirichlet_noise_still_valid():
    mcts = _make_mcts(action_space=9)
    np.random.seed(0)
    obs = torch.zeros(3, 3, 3)
    root = mcts.run(obs, list(range(9)), add_noise=True)
    assert root.visit_count == mcts.config.num_simulations
    assert np.isclose(root.child_priors.sum(), 1.0)


def test_mcts_run_respects_legal_actions_at_root():
    mcts = _make_mcts(action_space=9)
    legal = [0, 4, 8]
    obs = torch.zeros(3, 3, 3)
    root = mcts.run(obs, legal, add_noise=False)
    # Only legal actions get root children.
    assert sorted(root.child_actions.tolist()) == legal


# --- Integration: BatchedMCTS ----------------------------------------------

def test_batched_mcts_produces_n_valid_roots():
    mcts = BatchedMCTS(
        network=StubNetwork(9),
        game=StubGame(9),
        config=StubConfig(),
        device="cpu",
    )
    n = 3
    obs_list = [torch.zeros(3, 3, 3) for _ in range(n)]
    legal_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8] for _ in range(n)]
    roots = mcts.run_batch(obs_list, legal_list, add_noise=False)

    assert len(roots) == n
    for root in roots:
        assert root.visit_count == mcts.config.num_simulations
        assert int(root.child_visits.sum()) == mcts.config.num_simulations


def test_batched_mcts_each_game_respects_its_own_legals():
    mcts = BatchedMCTS(StubNetwork(9), StubGame(9), StubConfig(), device="cpu")
    obs_list = [torch.zeros(3, 3, 3) for _ in range(2)]
    legal_list = [[0, 1, 2], [6, 7, 8]]
    roots = mcts.run_batch(obs_list, legal_list, add_noise=False)
    assert sorted(roots[0].child_actions.tolist()) == [0, 1, 2]
    assert sorted(roots[1].child_actions.tolist()) == [6, 7, 8]
