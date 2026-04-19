"""Tests for Sampled MuZero — API + math contract.

Follows Hubert 2021 Proposed Modification: after sampling K actions σ from
softmax(logits), PUCT uses π_net renormalized over σ, and the training target
is raw `N(a)/ΣN(a)`. No post-hoc β̂/β correction at target time (that was
the Naive Modification the paper explicitly rejects).

Run with: pytest tests/test_sampled_muzero.py
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

# Guarded so collection still works on a stale tree; individual tests skip.
try:
    from src.mcts.sampling import sample_topk_gumbel
    _HAS_SAMPLING = True
except ImportError:
    _HAS_SAMPLING = False

from src.mcts.mcts import MCTS, BatchedMCTS, MCTSNode, MinMaxStats, select_action


# --- Shared fixtures for BatchedMCTS integration tests --------------------

@dataclass
class _StubGame:
    action_space_size: int = 9


@dataclass
class _StubConfig:
    discount: float = 1.0
    num_simulations: int = 8
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    leaf_top_k: int | None = None
    sample_k: int | None = None


class _StubNetwork:
    """Deterministic zero-logit stub — softmax is uniform, all actions tie."""

    def __init__(self, action_space: int = 9, hidden_dim: int = 4):
        self.action_space = action_space
        self.hidden_dim = hidden_dim

    def initial_inference(self, obs: torch.Tensor):
        b = obs.shape[0]
        return (
            torch.zeros(b, self.hidden_dim),
            torch.zeros(b, self.action_space),
            torch.zeros(b, 1),
        )

    def recurrent_inference(self, hidden: torch.Tensor, action: torch.Tensor):
        b = hidden.shape[0]
        return (
            torch.zeros(b, self.hidden_dim),
            torch.zeros(b, 1),
            torch.zeros(b, self.action_space),
            torch.zeros(b, 1),
        )


def _make_batched(action_space: int = 9, sample_k: int | None = None,
                  num_simulations: int = 8) -> BatchedMCTS:
    cfg = _StubConfig(sample_k=sample_k, num_simulations=num_simulations)
    return BatchedMCTS(
        network=_StubNetwork(action_space),
        game=_StubGame(action_space),
        config=cfg,
        device="cpu",
    )


# --- Gumbel-Top-K sampling utility ----------------------------------------

@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_returns_k_unique_actions():
    """Gumbel-Top-K must sample K distinct action indices (no replacement)."""
    logits = torch.tensor([1.0, 2.0, 0.5, -1.0, 3.0, 0.0, 2.5, 1.5, 0.8, 0.3])
    rng = np.random.default_rng(0)
    actions, _ = sample_topk_gumbel(logits, k=5, rng=rng)
    assert len(actions) == 5
    assert len(set(actions.tolist())) == 5


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_k_equals_n_returns_all_actions():
    """Degenerate case: K = number of actions → sample returns every action."""
    logits = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.0])
    rng = np.random.default_rng(1)
    actions, _ = sample_topk_gumbel(logits, k=5, rng=rng)
    assert sorted(actions.tolist()) == [0, 1, 2, 3, 4]


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_distribution_matches_prior_in_expectation():
    """Gumbel-Top-1 (k=1) is equivalent to sampling from softmax(logits).
    Over many samples, the empirical distribution should match softmax(logits)."""
    logits = torch.tensor([1.0, 2.0, 0.5, -1.0, 0.3])
    expected_probs = torch.softmax(logits, dim=0).numpy()
    rng = np.random.default_rng(42)
    counts = np.zeros(5)
    n_samples = 20000
    for _ in range(n_samples):
        actions, _ = sample_topk_gumbel(logits, k=1, rng=rng)
        counts[actions[0]] += 1
    empirical = counts / n_samples
    np.testing.assert_allclose(empirical, expected_probs, atol=0.02)


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_batched_shape():
    """Batched Gumbel-Top-K over (B, A) logits returns (B, K) actions."""
    logits = torch.randn(8, 20)
    rng = np.random.default_rng(0)
    actions, _ = sample_topk_gumbel(logits, k=5, rng=rng)
    assert actions.shape == (8, 5)
    for b in range(8):
        assert len(set(actions[b].tolist())) == 5


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_deterministic_under_same_seed():
    """Same rng seed must produce identical sampled indices — critical for repro."""
    logits = torch.tensor([0.5, 1.2, -0.3, 2.0, 0.1, -1.5, 0.8, 1.4])
    a1, p1 = sample_topk_gumbel(logits, k=3, rng=np.random.default_rng(123))
    a2, p2 = sample_topk_gumbel(logits, k=3, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_allclose(p1, p2)


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_different_seeds_give_different_samples():
    """Different seeds should (almost surely) give different orderings."""
    logits = torch.zeros(20)
    a1, _ = sample_topk_gumbel(logits, k=5, rng=np.random.default_rng(1))
    a2, _ = sample_topk_gumbel(logits, k=5, rng=np.random.default_rng(2))
    assert sorted(a1.tolist()) != sorted(a2.tolist()) or not np.array_equal(a1, a2)


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_perturbed_returned_in_descending_order():
    """Returned perturbed values should be sorted descending."""
    logits = torch.tensor([0.1, 2.0, 0.5, -1.0, 1.5, 0.3, 0.9])
    _, perturbed = sample_topk_gumbel(logits, k=4, rng=np.random.default_rng(7))
    assert np.all(np.diff(perturbed) <= 0)


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_dominant_logit_always_sampled():
    """A logit ~30 nats above the rest means softmax prob ≈ 1 — must always be picked."""
    logits = torch.tensor([0.0, 0.0, 30.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(100):
        actions, _ = sample_topk_gumbel(logits, k=1, rng=rng)
        assert actions[0] == 2


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_batched_rows_are_independent():
    """Different rows with different logit patterns should sample accordingly."""
    dominant = torch.full((4, 6), -10.0)
    dominant[0, 3] = 50.0
    dominant[1, 1] = 50.0
    dominant[2, 5] = 50.0
    dominant[3, 0] = 50.0
    actions, _ = sample_topk_gumbel(dominant, k=1, rng=np.random.default_rng(42))
    assert actions.shape == (4, 1)
    assert actions[0, 0] == 3
    assert actions[1, 0] == 1
    assert actions[2, 0] == 5
    assert actions[3, 0] == 0


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_accepts_numpy_input():
    """Utility must also accept a numpy array, not just torch."""
    logits = np.array([0.5, 1.2, -0.3, 2.0, 0.1], dtype=np.float64)
    actions, _ = sample_topk_gumbel(logits, k=3, rng=np.random.default_rng(0))
    assert actions.shape == (3,)
    assert len(set(actions.tolist())) == 3


# --- MCTSNode storage: single-prior field, no separate IS denominator -----

def test_mctsnode_has_no_child_priors_net_field():
    """Under the Proposed Modification we do not need a separate pre-noise net
    prior — the search prior itself is the Proposed-Modification prior π_net|σ."""
    node = MCTSNode()
    assert not hasattr(node, "child_priors_net"), \
        "child_priors_net belongs to the Naive variant we no longer implement"


def test_expand_from_priors_stores_priors():
    """_expand_from_priors sets child_priors = input priors (the PUCT prior)."""
    cfg = type("C", (), {"discount": 1.0})()
    game = type("G", (), {"action_space_size": 9})()
    mcts = MCTS(network=None, game=game, config=cfg, device="cpu")
    node = MCTSNode()
    actions = np.array([0, 1, 2], dtype=np.int64)
    priors = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    mcts._expand_from_priors(node, actions, priors)
    np.testing.assert_array_equal(node.child_priors, priors)


def test_dirichlet_noise_preserves_prior_mass():
    """After _add_dirichlet_noise: child_priors still sums to 1 (mixing two
    probability distributions stays a probability distribution)."""
    cfg = type("C", (), {
        "discount": 1.0, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
    })()
    game = type("G", (), {"action_space_size": 9})()
    mcts = MCTS(network=None, game=game, config=cfg, device="cpu")
    node = MCTSNode()
    actions = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    priors = np.array([0.2] * 5, dtype=np.float64)
    mcts._expand_from_priors(node, actions, priors)
    priors_before = node.child_priors.copy()
    np.random.seed(0)
    mcts._add_dirichlet_noise(node)
    assert not np.allclose(node.child_priors, priors_before), "noise must change priors"
    assert node.child_priors.sum() == pytest.approx(1.0), "post-noise priors must still sum to 1"


def test_dirichlet_noise_syncs_child_prior_to_search_prior():
    """After noise, each child.prior should reflect child_priors (used by PUCT)."""
    cfg = type("C", (), {
        "discount": 1.0, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
    })()
    game = type("G", (), {"action_space_size": 9})()
    mcts = MCTS(network=None, game=game, config=cfg, device="cpu")
    node = MCTSNode()
    actions = np.array([0, 1, 2], dtype=np.int64)
    priors = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    mcts._expand_from_priors(node, actions, priors)
    np.random.seed(0)
    mcts._add_dirichlet_noise(node)
    for i, child in enumerate(node.children):
        assert child.prior == pytest.approx(float(node.child_priors[i]))


# --- Policy target: raw normalized visit counts ---------------------------

def test_policy_target_is_raw_visit_normalization():
    """Under the Proposed Modification the target is N(a)/ΣN(a) — raw visits.
    The β̂/β correction lives inside the search prior, not at target time."""
    root = MCTSNode()
    root.child_actions = np.array([1, 3, 5], dtype=np.int64)
    root.child_visits = np.array([10.0, 20.0, 5.0], dtype=np.float64)
    root.children = [MCTSNode(visit_count=int(v)) for v in root.child_visits]

    _, probs = select_action(root, temperature=1.0)
    probs = np.asarray(probs)

    total = 35.0
    assert probs[1] == pytest.approx(10 / total)
    assert probs[3] == pytest.approx(20 / total)
    assert probs[5] == pytest.approx(5 / total)
    # Unsampled actions → 0.
    assert probs[0] == 0.0
    assert probs[2] == 0.0
    assert probs[4] == 0.0


def test_policy_target_sums_to_one_over_sampled():
    """Raw visit target must be a valid probability distribution."""
    root = MCTSNode()
    root.child_actions = np.array([2, 5, 7, 9], dtype=np.int64)
    root.child_visits = np.array([5.0, 10.0, 3.0, 7.0], dtype=np.float64)
    root.children = [MCTSNode(visit_count=int(v)) for v in root.child_visits]

    _, probs = select_action(root, temperature=1.0)
    assert sum(probs) == pytest.approx(1.0)


def test_policy_target_zero_visits_falls_back_to_uniform():
    """Edge case: if MCTS didn't execute (all visits 0), fall back to uniform."""
    root = MCTSNode()
    root.child_actions = np.array([3, 6, 8], dtype=np.int64)
    root.child_visits = np.zeros(3, dtype=np.float64)
    root.children = [MCTSNode() for _ in range(3)]
    _, probs = select_action(root, temperature=1.0)
    probs = np.asarray(probs)
    assert probs[3] == pytest.approx(1 / 3)
    assert probs[6] == pytest.approx(1 / 3)
    assert probs[8] == pytest.approx(1 / 3)
    assert np.isfinite(probs).all()


def test_policy_target_temperature_zero_is_one_hot_argmax_visits():
    """Temperature=0: action = argmax(visits), policy = one-hot on that action."""
    root = MCTSNode()
    root.child_actions = np.array([0, 1, 2], dtype=np.int64)
    root.child_visits = np.array([2.0, 5.0, 1.0], dtype=np.float64)
    action, probs = select_action(root, temperature=0)
    assert action == 1
    probs = np.asarray(probs)
    assert probs[1] == pytest.approx(1.0)
    assert probs[0] == 0.0
    assert probs[2] == 0.0


def test_policy_target_single_child_is_deterministic():
    """K=1: only one sampled action, all probability mass on it."""
    root = MCTSNode()
    root.child_actions = np.array([4], dtype=np.int64)
    root.child_visits = np.array([7.0], dtype=np.float64)
    root.children = [MCTSNode(visit_count=7)]
    action, probs = select_action(root, temperature=1.0)
    assert action == 4
    assert probs[4] == pytest.approx(1.0)
    assert sum(probs) == pytest.approx(1.0)


def test_policy_target_permutation_invariant_over_storage_order():
    """Reordering sampled actions (and their aligned visits) must give the
    same distribution — the math depends on (action, visit) tuples."""
    r1 = MCTSNode()
    r1.child_actions = np.array([0, 1, 2], dtype=np.int64)
    r1.child_visits = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    r1.children = [MCTSNode(visit_count=int(v)) for v in r1.child_visits]
    r2 = MCTSNode()
    r2.child_actions = np.array([2, 0, 1], dtype=np.int64)
    r2.child_visits = np.array([15.0, 5.0, 10.0], dtype=np.float64)
    r2.children = [MCTSNode(visit_count=int(v)) for v in r2.child_visits]
    _, p1 = select_action(r1, temperature=1.0)
    _, p2 = select_action(r2, temperature=1.0)
    np.testing.assert_allclose(p1, p2)


# --- Parity: sample_k=None falls back to current behavior ----------------

def test_sample_k_config_field_exists():
    """MuZeroConfig needs sample_k field."""
    from src.config import MuZeroConfig
    cfg = MuZeroConfig()
    assert hasattr(cfg, "sample_k"), \
        "MuZeroConfig needs sample_k field (replaces leaf_top_k)"


# --- BatchedMCTS integration: sample_k at root and leaves ------------------

def test_batched_mcts_sample_k_none_expands_all_legals_at_root():
    """Default path: sample_k=None, all legal actions expanded at root."""
    mcts = _make_batched(action_space=9, sample_k=None, num_simulations=4)
    obs = [torch.zeros(3, 3, 3) for _ in range(2)]
    legals = [list(range(9)), [0, 2, 4, 6, 8]]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    assert len(roots[0].child_actions) == 9
    assert len(roots[1].child_actions) == 5
    for child in roots[0].children:
        if child.child_actions is not None:
            assert len(child.child_actions) == 9
            break


def test_batched_mcts_sample_k_larger_than_legals_is_no_op_at_root():
    """If K >= |legals| at root, all legals are expanded."""
    mcts = _make_batched(action_space=9, sample_k=20, num_simulations=4)
    obs = [torch.zeros(3, 3, 3)]
    legals = [[1, 3, 5, 7]]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    assert sorted(roots[0].child_actions.tolist()) == [1, 3, 5, 7]


def test_batched_mcts_sample_k_smaller_than_legals_samples_k_at_root():
    """When K < |legals|, root expands exactly K distinct legal actions."""
    mcts = _make_batched(action_space=9, sample_k=3, num_simulations=4)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    assert len(roots[0].child_actions) == 3
    assert len(set(roots[0].child_actions.tolist())) == 3
    for a in roots[0].child_actions:
        assert a in set(legals[0])


def test_batched_mcts_sample_k_applies_at_leaves_too():
    """Leaves always use K sampled children when sample_k is set."""
    mcts = _make_batched(action_space=9, sample_k=4, num_simulations=8)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    found = False
    for child in roots[0].children:
        if child.child_actions is not None:
            assert len(child.child_actions) == 4
            assert len(set(child.child_actions.tolist())) == 4
            found = True
            break
    assert found, "expected at least one leaf to have been expanded"


def test_batched_mcts_sampled_priors_are_renormalized_over_sigma():
    """Proposed Modification contract: after sampling, PUCT prior over σ
    must sum to 1 (π_net renormalized over the sampled subspace).
    This is the code path the paper's Theorem 1 actually prescribes."""
    mcts = _make_batched(action_space=9, sample_k=4, num_simulations=4)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    # add_noise=False so we inspect the renormalized-over-σ priors directly,
    # before Dirichlet mixing would obscure the invariant at the root.
    roots = mcts.run_batch(obs, legals, add_noise=False)
    root = roots[0]
    assert root.child_priors.sum() == pytest.approx(1.0), \
        "root priors over σ must sum to 1 (Hubert 2021 Proposed Modification)"
    assert np.all(root.child_priors > 0)
    # Check a leaf too.
    for child in root.children:
        if child.child_priors is not None and len(child.child_priors) == 4:
            assert child.child_priors.sum() == pytest.approx(1.0), \
                "leaf priors over σ must sum to 1"
            assert np.all(child.child_priors > 0)
            break


def test_batched_mcts_sample_k_none_uses_full_legal_softmax():
    """sample_k=None + no noise → root priors = legal-masked softmax over A_legal
    (sums to 1 over legal actions, same as pre-sampled-MuZero behavior)."""
    mcts = _make_batched(action_space=9, sample_k=None, num_simulations=2)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    assert roots[0].child_priors.sum() == pytest.approx(1.0)
    # Uniform logits → uniform softmax over all 9 legal actions.
    np.testing.assert_allclose(roots[0].child_priors, np.full(9, 1 / 9))
