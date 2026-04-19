"""Tests for Sampled MuZero — API + math contract.

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

@pytest.mark.skipif(not _HAS_SAMPLING, reason="Phase 2.1 not implemented yet")
def test_gumbel_topk_returns_k_unique_actions():
    """Gumbel-Top-K must sample K distinct action indices (no replacement)."""
    logits = torch.tensor([1.0, 2.0, 0.5, -1.0, 3.0, 0.0, 2.5, 1.5, 0.8, 0.3])
    rng = np.random.default_rng(0)
    actions, _ = sample_topk_gumbel(logits, k=5, rng=rng)
    assert len(actions) == 5
    assert len(set(actions.tolist())) == 5  # all unique


@pytest.mark.skipif(not _HAS_SAMPLING, reason="Phase 2.1 not implemented yet")
def test_gumbel_topk_k_equals_n_returns_all_actions():
    """Degenerate case: K = number of actions → sample returns every action."""
    logits = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.0])
    rng = np.random.default_rng(1)
    actions, _ = sample_topk_gumbel(logits, k=5, rng=rng)
    assert sorted(actions.tolist()) == [0, 1, 2, 3, 4]


@pytest.mark.skipif(not _HAS_SAMPLING, reason="Phase 2.1 not implemented yet")
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
    # Each component should match softmax within ~2% absolute error.
    np.testing.assert_allclose(empirical, expected_probs, atol=0.02)


@pytest.mark.skipif(not _HAS_SAMPLING, reason="Phase 2.1 not implemented yet")
def test_gumbel_topk_batched_shape():
    """Batched Gumbel-Top-K over (B, A) logits returns (B, K) actions."""
    logits = torch.randn(8, 20)  # batch of 8, 20 actions each
    rng = np.random.default_rng(0)
    actions, _ = sample_topk_gumbel(logits, k=5, rng=rng)
    assert actions.shape == (8, 5)
    # Each row has unique actions.
    for b in range(8):
        assert len(set(actions[b].tolist())) == 5


# --- MCTSNode storage: pre-noise prior separate from search prior ---------

def test_mctsnode_has_child_priors_net_field():
    """MCTSNode must carry pre-noise network priors separately from search priors."""
    node = MCTSNode()
    assert hasattr(node, "child_priors_net"), \
        "MCTSNode.child_priors_net required for Sampled MuZero IS correction"


def test_expand_from_priors_stores_priors_net():
    """_expand_from_priors should set child_priors_net = input priors.
    Initially child_priors == child_priors_net (noise only applied at root later)."""
    cfg = type("C", (), {"discount": 1.0})()
    game = type("G", (), {"action_space_size": 9})()
    mcts = MCTS(network=None, game=game, config=cfg, device="cpu")
    node = MCTSNode()
    actions = np.array([0, 1, 2], dtype=np.int64)
    priors = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    mcts._expand_from_priors(node, actions, priors)
    assert node.child_priors_net is not None
    np.testing.assert_array_equal(node.child_priors_net, priors)
    # Before any noise, search priors == net priors.
    np.testing.assert_array_equal(node.child_priors, node.child_priors_net)


def test_dirichlet_noise_mutates_search_priors_only():
    """After _add_dirichlet_noise: child_priors changes, child_priors_net does NOT.
    This keeps the pre-noise estimate intact for IS correction."""
    cfg = type("C", (), {
        "discount": 1.0, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
    })()
    game = type("G", (), {"action_space_size": 9})()
    mcts = MCTS(network=None, game=game, config=cfg, device="cpu")
    node = MCTSNode()
    actions = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    priors = np.array([0.2] * 5, dtype=np.float64)
    mcts._expand_from_priors(node, actions, priors)
    priors_net_before = node.child_priors_net.copy()
    np.random.seed(0)
    mcts._add_dirichlet_noise(node)
    # search priors must have changed
    assert not np.allclose(node.child_priors, priors_net_before)
    # network priors must NOT have changed
    np.testing.assert_array_equal(node.child_priors_net, priors_net_before)


def test_dirichlet_noise_syncs_child_prior_to_search_not_net():
    """After noise, each child.prior should reflect the search-time mixed prior
    (used by PUCT). child_priors_net is for IS math only and not per-child."""
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


# --- IS-corrected policy target math --------------------------------------

def test_importance_corrected_policy_hand_computed():
    """Sampled MuZero IS-corrected policy:

        π_target(a) ∝ N(a) / π_net(a)    for sampled a
        π_target(a) = 0                   for unsampled a

    Example: sampled actions [1, 3, 5] with priors [0.1, 0.2, 0.05] and
    visit counts [10, 20, 5]. Weights: 100, 100, 100 → uniform over sampled.
    """
    root = MCTSNode()
    root.child_actions = np.array([1, 3, 5], dtype=np.int64)
    root.child_priors_net = np.array([0.1, 0.2, 0.05], dtype=np.float64)
    root.child_visits = np.array([10.0, 20.0, 5.0], dtype=np.float64)
    # Build children so select_action can read visit counts in the usual way.
    root.children = [MCTSNode(visit_count=int(v)) for v in root.child_visits]

    action, probs = select_action(root, temperature=1.0, use_importance_sampling=True)
    probs = np.asarray(probs)

    # Weights are all 100 → uniform over sampled actions {1, 3, 5}.
    assert probs[1] == pytest.approx(1 / 3)
    assert probs[3] == pytest.approx(1 / 3)
    assert probs[5] == pytest.approx(1 / 3)
    # Unsampled actions (0, 2, 4) should be 0.
    assert probs[0] == 0.0
    assert probs[2] == 0.0
    assert probs[4] == 0.0


def test_importance_corrected_policy_low_prior_high_visits_gets_mass():
    """A low-prior action with high visits should dominate the IS target.
    This is the behavioral win that Sampled MuZero provides over top-K."""
    root = MCTSNode()
    root.child_actions = np.array([0, 1], dtype=np.int64)
    # Action 0: low prior 0.01, high visits 10 → weight 1000
    # Action 1: high prior 0.5, 20 visits  → weight   40
    root.child_priors_net = np.array([0.01, 0.5], dtype=np.float64)
    root.child_visits = np.array([10.0, 20.0], dtype=np.float64)
    root.children = [MCTSNode(visit_count=10), MCTSNode(visit_count=20)]

    _, probs = select_action(root, temperature=1.0, use_importance_sampling=True)
    probs = np.asarray(probs)
    # Action 0 should carry ~1000/(1000+40) ≈ 0.9615 of the mass.
    assert probs[0] == pytest.approx(1000 / 1040, rel=1e-4)
    assert probs[1] == pytest.approx(40 / 1040, rel=1e-4)


def test_importance_corrected_policy_sums_to_one_over_sampled():
    """IS target must be a valid probability distribution over sampled actions."""
    root = MCTSNode()
    root.child_actions = np.array([2, 5, 7, 9], dtype=np.int64)
    root.child_priors_net = np.array([0.1, 0.3, 0.05, 0.2], dtype=np.float64)
    root.child_visits = np.array([5.0, 10.0, 3.0, 7.0], dtype=np.float64)
    root.children = [MCTSNode(visit_count=int(v)) for v in root.child_visits]

    _, probs = select_action(root, temperature=1.0, use_importance_sampling=True)
    assert sum(probs) == pytest.approx(1.0)


def test_select_action_default_still_uses_raw_visits():
    """Default select_action (use_importance_sampling=False) must still match
    current behavior — agent's move choice stays PUCT+visit-count based."""
    root = MCTSNode()
    root.child_actions = np.array([0, 1, 2], dtype=np.int64)
    root.child_visits = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    root.child_priors_net = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    _, probs = select_action(root, temperature=1.0)
    probs = np.asarray(probs)
    # Default: visits-proportional, priors ignored.
    assert probs[0] == pytest.approx(1 / 6)
    assert probs[1] == pytest.approx(2 / 6)
    assert probs[2] == pytest.approx(3 / 6)


# --- Parity: sample_k=None falls back to current behavior ----------------

def test_sample_k_none_is_degenerate_case():
    """With sample_k=None in config, BatchedMCTS should produce the same
    trees as the current implementation — IS is opt-in, not forced."""
    # This is a smoke-level parity test; more detail lives in integration tests.
    # At 2.0 it fails because the config flag doesn't exist yet.
    from src.config import MuZeroConfig
    cfg = MuZeroConfig()
    assert hasattr(cfg, "sample_k"), \
        "MuZeroConfig needs sample_k field (replaces leaf_top_k)"


# --- Gumbel-Top-K: determinism, edge cases, batched parity ---------------

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
    """Different seeds should (almost surely) give different orderings for uniform logits."""
    logits = torch.zeros(20)  # perfectly uniform softmax
    a1, _ = sample_topk_gumbel(logits, k=5, rng=np.random.default_rng(1))
    a2, _ = sample_topk_gumbel(logits, k=5, rng=np.random.default_rng(2))
    # Extremely unlikely that two independent draws of 5/20 produce identical sorted sets.
    assert sorted(a1.tolist()) != sorted(a2.tolist()) or not np.array_equal(a1, a2)


@pytest.mark.skipif(not _HAS_SAMPLING, reason="sampling module missing")
def test_gumbel_topk_perturbed_returned_in_descending_order():
    """Returned perturbed values should be sorted descending so downstream code
    can treat index 0 as the strongest sample."""
    logits = torch.tensor([0.1, 2.0, 0.5, -1.0, 1.5, 0.3, 0.9])
    _, perturbed = sample_topk_gumbel(logits, k=4, rng=np.random.default_rng(7))
    assert np.all(np.diff(perturbed) <= 0), "perturbed values must be non-increasing"


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
    """Different rows with different logit patterns should sample accordingly —
    batched call must not mix samples across rows."""
    dominant = torch.full((4, 6), -10.0)
    dominant[0, 3] = 50.0  # row 0: action 3 dominant
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
    """Utility must also accept a numpy array, not just torch — keeps BatchedMCTS
    free of an unnecessary tensor→tensor→numpy roundtrip at the root."""
    logits = np.array([0.5, 1.2, -0.3, 2.0, 0.1], dtype=np.float64)
    actions, _ = sample_topk_gumbel(logits, k=3, rng=np.random.default_rng(0))
    assert actions.shape == (3,)
    assert len(set(actions.tolist())) == 3


# --- BatchedMCTS integration: sample_k at root and leaves ------------------

def test_batched_mcts_sample_k_none_expands_all_legals_at_root():
    """Default path: sample_k=None, all legal actions expanded at root."""
    mcts = _make_batched(action_space=9, sample_k=None, num_simulations=4)
    obs = [torch.zeros(3, 3, 3) for _ in range(2)]
    legals = [list(range(9)), [0, 2, 4, 6, 8]]  # full, 5-legal
    roots = mcts.run_batch(obs, legals, add_noise=False)
    assert len(roots[0].child_actions) == 9
    assert len(roots[1].child_actions) == 5
    # Leaves: no leaf_top_k either → expand all action_space_size.
    # Pick a child that got visited to confirm leaf expansion width.
    for child in roots[0].children:
        if child.child_actions is not None:
            assert len(child.child_actions) == 9
            break


def test_batched_mcts_sample_k_larger_than_legals_is_no_op_at_root():
    """If K >= |legals| at root, all legals are expanded (Gumbel-Top-K degenerate)."""
    mcts = _make_batched(action_space=9, sample_k=20, num_simulations=4)
    obs = [torch.zeros(3, 3, 3)]
    legals = [[1, 3, 5, 7]]  # only 4 legal moves
    roots = mcts.run_batch(obs, legals, add_noise=False)
    assert sorted(roots[0].child_actions.tolist()) == [1, 3, 5, 7]


def test_batched_mcts_sample_k_smaller_than_legals_samples_k_at_root():
    """When K < |legals|, root expands exactly K distinct legal actions."""
    mcts = _make_batched(action_space=9, sample_k=3, num_simulations=4)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]  # all 9 legal
    roots = mcts.run_batch(obs, legals, add_noise=False)
    assert len(roots[0].child_actions) == 3
    assert len(set(roots[0].child_actions.tolist())) == 3
    # All sampled actions must be from the legal set.
    for a in roots[0].child_actions:
        assert a in set(legals[0])


def test_batched_mcts_sample_k_applies_at_leaves_too():
    """Leaves (latent space) always use K sampled children when sample_k is set.
    No legality mask at leaves — domain is the full action space."""
    mcts = _make_batched(action_space=9, sample_k=4, num_simulations=8)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    # Find any expanded leaf and check it has 4 sampled children.
    found = False
    for child in roots[0].children:
        if child.child_actions is not None:
            assert len(child.child_actions) == 4
            assert len(set(child.child_actions.tolist())) == 4
            found = True
            break
    assert found, "expected at least one leaf to have been expanded"


def test_batched_mcts_populates_child_priors_net_at_root_and_leaves():
    """child_priors_net must be populated by run_batch — the IS training target
    computation depends on it."""
    mcts = _make_batched(action_space=9, sample_k=4, num_simulations=4)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    roots = mcts.run_batch(obs, legals, add_noise=True)
    root = roots[0]
    assert root.child_priors_net is not None
    assert root.child_priors_net.shape == root.child_priors.shape
    # Dirichlet touched search priors but not the net priors.
    assert not np.allclose(root.child_priors, root.child_priors_net)
    # Net priors must be a valid softmax-subset: positive, each ≤ 1.
    assert np.all(root.child_priors_net > 0)
    assert np.all(root.child_priors_net <= 1.0)
    # Check a leaf too.
    for child in root.children:
        if child.child_priors_net is not None:
            assert child.child_priors_net.shape == (4,)
            assert np.all(child.child_priors_net > 0)
            break


def test_batched_mcts_sample_k_none_does_not_populate_gumbel_noise():
    """sample_k=None → root uses full legal expansion; child_priors equals
    child_priors_net initially (ignoring Dirichlet)."""
    mcts = _make_batched(action_space=9, sample_k=None, num_simulations=2)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    # No noise → search priors should equal net priors.
    np.testing.assert_array_equal(roots[0].child_priors, roots[0].child_priors_net)


# --- select_action IS interactions -----------------------------------------

def test_select_action_is_with_temperature_zero_picks_argmax_visits_but_returns_is_probs():
    """IS affects the *policy target*, not the *move* choice.
    Temperature=0 must return argmax-visits action while probs stay IS-corrected."""
    root = MCTSNode()
    root.child_actions = np.array([0, 1, 2], dtype=np.int64)
    root.child_priors_net = np.array([0.5, 0.4, 0.1], dtype=np.float64)
    root.child_visits = np.array([2.0, 5.0, 1.0], dtype=np.float64)
    action, probs = select_action(root, temperature=0, use_importance_sampling=True)
    # argmax visits → action 1
    assert action == 1
    # IS-corrected probs: weights = [4, 12.5, 10] → sum 26.5
    probs = np.asarray(probs)
    assert probs[0] == pytest.approx(4 / 26.5)
    assert probs[1] == pytest.approx(12.5 / 26.5)
    assert probs[2] == pytest.approx(10 / 26.5)


def test_select_action_is_zero_visits_falls_back_to_uniform():
    """Edge case: if MCTS didn't execute (all visits 0), IS target has total weight 0.
    Must fall back to uniform over sampled, not produce nan."""
    root = MCTSNode()
    root.child_actions = np.array([3, 6, 8], dtype=np.int64)
    root.child_priors_net = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    root.child_visits = np.zeros(3, dtype=np.float64)
    _, probs = select_action(root, temperature=1.0, use_importance_sampling=True)
    probs = np.asarray(probs)
    assert probs[3] == pytest.approx(1 / 3)
    assert probs[6] == pytest.approx(1 / 3)
    assert probs[8] == pytest.approx(1 / 3)
    assert np.isfinite(probs).all()


def test_select_action_is_tiny_prior_does_not_produce_nan_or_inf():
    """Numerical guard: a prior that's numerically zero must not cause divide-by-zero."""
    root = MCTSNode()
    root.child_actions = np.array([0, 1], dtype=np.int64)
    root.child_priors_net = np.array([1e-20, 0.5], dtype=np.float64)  # effectively zero
    root.child_visits = np.array([1.0, 1.0], dtype=np.float64)
    _, probs = select_action(root, temperature=1.0, use_importance_sampling=True)
    probs = np.asarray(probs)
    assert np.isfinite(probs).all()
    assert sum(probs) == pytest.approx(1.0)
    # The tiny-prior action should dominate — weight 1/1e-12 vs 1/0.5.
    assert probs[0] > probs[1]


def test_select_action_is_single_child_returns_deterministic_prob_one():
    """K=1: only one sampled action, all probability mass on it."""
    root = MCTSNode()
    root.child_actions = np.array([4], dtype=np.int64)
    root.child_priors_net = np.array([0.1], dtype=np.float64)
    root.child_visits = np.array([7.0], dtype=np.float64)
    action, probs = select_action(root, temperature=1.0, use_importance_sampling=True)
    assert action == 4
    assert probs[4] == pytest.approx(1.0)
    assert sum(probs) == pytest.approx(1.0)


def test_select_action_is_permutation_invariant_over_sampled_order():
    """Reordering sampled actions (and their aligned priors/visits) must give the
    same IS-corrected distribution — math depends on the (action, prior, visit) tuples,
    not on storage order."""
    # Base ordering.
    r1 = MCTSNode()
    r1.child_actions = np.array([0, 1, 2], dtype=np.int64)
    r1.child_priors_net = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    r1.child_visits = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    # Permuted ordering.
    r2 = MCTSNode()
    r2.child_actions = np.array([2, 0, 1], dtype=np.int64)
    r2.child_priors_net = np.array([0.3, 0.1, 0.2], dtype=np.float64)
    r2.child_visits = np.array([15.0, 5.0, 10.0], dtype=np.float64)
    _, p1 = select_action(r1, temperature=1.0, use_importance_sampling=True)
    _, p2 = select_action(r2, temperature=1.0, use_importance_sampling=True)
    np.testing.assert_allclose(p1, p2)
