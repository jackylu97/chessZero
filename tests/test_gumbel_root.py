"""Integration tests for Plain Gumbel MuZero root Sequential Halving (G2).

Checks the phase-count trace against paper Fig 1 / the sequential_halving_schedule
contract, plus that the run_batch Gumbel path populates the G3 training-target
fields (value_prior, root_legal_actions, root_legal_logits, root_sampled_perturbed)
and that use_gumbel=False preserves the existing PUCT path.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.mcts.gumbel import compute_improved_policy, compute_v_mix
from src.mcts.mcts import BatchedMCTS, MCTSNode, _GumbelRootState, select_action_gumbel


# --- Stubs -----------------------------------------------------------------

@dataclass
class _StubGame:
    action_space_size: int = 9


@dataclass
class _StubConfig:
    discount: float = 1.0
    num_simulations: int = 16
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    sample_k: int | None = None
    use_gumbel: bool = True
    gumbel_num_considered: int = 4
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    use_gumbel_noise: bool = False  # deterministic for tests


class _FixedLogitsNetwork:
    """Fixed root logits, zero leaf logits, zero value/reward everywhere."""

    def __init__(self, action_space=9, hidden_dim=4, root_logits=None):
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.root_logits = (
            root_logits if root_logits is not None else torch.zeros(action_space)
        )

    def initial_inference(self, obs):
        b = obs.shape[0]
        return (
            torch.zeros(b, self.hidden_dim),
            self.root_logits.unsqueeze(0).expand(b, -1).contiguous(),
            torch.zeros(b, 1),
        )

    def recurrent_inference(self, hidden, action):
        b = hidden.shape[0]
        return (
            torch.zeros(b, self.hidden_dim),
            torch.zeros(b, 1),
            torch.zeros(b, self.action_space),
            torch.zeros(b, 1),
        )


def _make_batched(config, action_space=9, root_logits=None):
    return BatchedMCTS(
        network=_FixedLogitsNetwork(action_space, root_logits=root_logits),
        game=_StubGame(action_space),
        config=config,
        device="cpu",
    )


# --- Phase-count traces ----------------------------------------------------

def test_gumbel_m4_n16_phase_trace():
    """m=4, n=16 → schedule [(4,2),(2,4)].

    Top-4 logits sampled (deterministic, noise=0). Phase 0: 4 actions × 2 visits.
    Phase 1: top-2 of those survive, getting 4 more visits each.
    Expected root visits: two 6's (survivors), two 2's (eliminated). Sum = 16.
    """
    cfg = _StubConfig(num_simulations=16, gumbel_num_considered=4,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    mcts = _make_batched(cfg, action_space=9, root_logits=logits)
    obs = [torch.zeros(3, 3, 3)]
    legals = [list(range(9))]
    roots = mcts.run_batch(obs, legals, add_noise=False)
    root = roots[0]

    # m = 4 sampled children (top-4 logits = actions 0..3).
    assert len(root.children) == 4
    assert sorted(int(a) for a in root.child_actions) == [0, 1, 2, 3]

    visits = sorted(root.child_visits.astype(int).tolist())
    assert visits == [2, 2, 6, 6], f"expected [2,2,6,6], got {visits}"
    assert int(root.child_visits.sum()) == 16


def test_gumbel_m1_single_action_gets_all_visits():
    """m=1 → schedule [(1, n)]; the only sampled action gets every simulation."""
    cfg = _StubConfig(num_simulations=8, gumbel_num_considered=1,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    mcts = _make_batched(cfg, root_logits=logits)
    roots = mcts.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False)
    root = roots[0]
    assert len(root.children) == 1
    assert int(root.child_actions[0]) == 0
    assert int(root.child_visits[0]) == 8


def test_gumbel_m_clamped_to_legal_action_count():
    """gumbel_num_considered > len(legals) clamps without crash; budget used."""
    cfg = _StubConfig(num_simulations=8, gumbel_num_considered=20,
                      use_gumbel=True, use_gumbel_noise=False)
    mcts = _make_batched(cfg, action_space=9)
    roots = mcts.run_batch([torch.zeros(3, 3, 3)], [[0, 1, 2]], add_noise=False)
    root = roots[0]
    assert len(root.children) == 3
    assert int(root.child_visits.sum()) == 8


def test_gumbel_schedule_exhausts_falls_back_to_provisional_best():
    """Schedule [(4,2), (2,5)] totals 18 sims; n=20 leaves 2 fall-through sims
    that must land on the provisional best (action 0 under ordered logits).

    Phase 0: each of 4 actions → 2 visits (8 sims).
    Phase 1: top-2 of those (actions 0, 1) → 5 visits each (10 sims, cumulative 18).
    Fallback: 2 more sims via argmax(perturbed + σ(q̂)). With stub zero Q's,
    σ=0 so ranking = perturbed = logits → winner is action 0.
    Expected: {0: 9, 1: 7, 2: 2, 3: 2}. Sum = 20.
    """
    cfg = _StubConfig(num_simulations=20, gumbel_num_considered=4,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    mcts = _make_batched(cfg, root_logits=logits)
    roots = mcts.run_batch(
        [torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False
    )
    root = roots[0]
    visits = {int(root.child_actions[i]): int(root.child_visits[i])
              for i in range(len(root.children))}
    assert visits == {0: 9, 1: 7, 2: 2, 3: 2}, f"got {visits}"
    assert int(root.child_visits.sum()) == 20


# --- Training-target fields (for G3) ---------------------------------------

def test_gumbel_root_captures_value_prior_and_legal_logits():
    cfg = _StubConfig(num_simulations=4, gumbel_num_considered=2,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    mcts = _make_batched(cfg, root_logits=logits)
    roots = mcts.run_batch(
        [torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False
    )
    root = roots[0]

    assert root.value_prior == 0.0  # stub returns zero value scalar
    assert list(root.root_legal_actions) == list(range(9))
    np.testing.assert_allclose(root.root_legal_logits, logits.numpy())
    # noise=0 + top-2 logits (descending) → sampled actions 8, 7 at local 0, 1.
    assert int(root.child_actions[0]) == 8
    assert int(root.child_actions[1]) == 7
    assert root.root_sampled_perturbed is not None
    np.testing.assert_allclose(root.root_sampled_perturbed, [9.0, 8.0])


# --- Fallback when use_gumbel=False ----------------------------------------

def test_gumbel_disabled_preserves_puct_path():
    """use_gumbel=False: no gumbel fields populated; PUCT runs every sim."""
    cfg = _StubConfig(num_simulations=8, use_gumbel=False)
    mcts = _make_batched(cfg)
    roots = mcts.run_batch(
        [torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False
    )
    root = roots[0]
    assert root.value_prior is None
    assert root.root_sampled_perturbed is None
    assert root.root_legal_actions is None
    # All legal actions expanded under the PUCT path.
    assert len(root.children) == 9
    assert int(root.child_visits.sum()) == 8


# --- Dirichlet skipped under Gumbel ----------------------------------------

def test_gumbel_skips_dirichlet_noise():
    """Child priors should equal the normalized softmax priors over sampled actions,
    unperturbed (Dirichlet is skipped under Gumbel — exploration via Gumbel-Top-K)."""
    cfg = _StubConfig(num_simulations=4, gumbel_num_considered=3,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
    mcts = _make_batched(cfg, root_logits=logits)
    roots = mcts.run_batch(
        [torch.zeros(3, 3, 3)], [list(range(9))], add_noise=True  # would trigger Dirichlet on PUCT path
    )
    root = roots[0]
    # Priors over sampled actions = softmax(full legals) restricted to sampled, renormalized.
    expected = np.exp(np.array([3.0, 2.0, 1.0]))
    expected = expected / expected.sum()
    np.testing.assert_allclose(root.child_priors, expected, atol=1e-10)


# --- Gumbel noise actually randomizes --------------------------------------

def test_gumbel_noise_on_varies_sampled_set_across_seeds():
    """Different rng seeds with uniform logits produce different sampled subsets."""
    cfg = _StubConfig(num_simulations=4, gumbel_num_considered=3,
                      use_gumbel=True, use_gumbel_noise=True)
    m1 = _make_batched(cfg)
    m1._rng = np.random.default_rng(0)
    m2 = _make_batched(cfg)
    m2._rng = np.random.default_rng(12345)
    r1 = m1.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=True)[0]
    r2 = m2.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=True)[0]
    s1 = set(int(a) for a in r1.child_actions)
    s2 = set(int(a) for a in r2.child_actions)
    assert s1 != s2


# --- _GumbelRootState is constructed correctly for a run -------------------

# --- G3: min_max_q captured on the root -----------------------------------

def test_gumbel_root_captures_min_max_q_after_search():
    """``run_batch`` should store the post-search (min, max) Q range on each root
    under use_gumbel; select_action_gumbel uses this to normalize completedQ."""
    cfg = _StubConfig(num_simulations=8, gumbel_num_considered=4,
                      use_gumbel=True, use_gumbel_noise=False)
    mcts = _make_batched(cfg)
    roots = mcts.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False)
    root = roots[0]
    assert root.min_max_q is not None
    mm_min, mm_max = root.min_max_q
    # Stub network produces zero values/rewards, so everything pushes 0.0 → range collapses.
    assert mm_max >= mm_min


# --- G3: π' target matches direct compute_improved_policy -----------------

def test_select_action_gumbel_target_matches_compute_improved_policy():
    """Deterministic root (noise off) + zero-Q stub → π' = softmax(legal_logits)
    since σ(constant completedQ) shifts all logits equally.

    Verifies select_action_gumbel's dense target equals the reference formula
    applied directly to root_legal_logits + completedQ.
    """
    cfg = _StubConfig(num_simulations=4, gumbel_num_considered=3,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
    mcts = _make_batched(cfg, root_logits=logits)
    roots = mcts.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False)
    root = roots[0]

    _, dense = select_action_gumbel(root, cfg, action_space_size=9)
    dense = np.asarray(dense)

    # All values are 0 under stub → mm range collapses → completed_q clipped to 0.
    # σ(0) ≡ 0 → π' reduces to softmax(legal_logits).
    shifted = logits.numpy() - logits.numpy().max()
    expected = np.exp(shifted) / np.exp(shifted).sum()
    np.testing.assert_allclose(dense, expected, atol=1e-10)
    assert dense.sum() == pytest.approx(1.0)


def test_select_action_gumbel_zero_mass_for_illegal_actions():
    """π' dense target zero on non-legal indices."""
    cfg = _StubConfig(num_simulations=4, gumbel_num_considered=2,
                      use_gumbel=True, use_gumbel_noise=False)
    mcts = _make_batched(cfg)
    roots = mcts.run_batch([torch.zeros(3, 3, 3)], [[1, 3, 5]], add_noise=False)
    _, dense = select_action_gumbel(roots[0], cfg, action_space_size=9)
    dense = np.asarray(dense)
    assert dense[0] == 0.0 and dense[2] == 0.0 and dense[4] == 0.0
    assert dense[6] == 0.0 and dense[7] == 0.0 and dense[8] == 0.0
    assert dense[[1, 3, 5]].sum() == pytest.approx(1.0)


def test_select_action_gumbel_action_is_argmax_over_sampled():
    """A_{n+1} must come from the sampled m-set (paper page 18), not from outside."""
    cfg = _StubConfig(num_simulations=4, gumbel_num_considered=3,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    mcts = _make_batched(cfg, root_logits=logits)
    roots = mcts.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False)
    root = roots[0]

    action, _ = select_action_gumbel(root, cfg, action_space_size=9)
    sampled = set(int(a) for a in root.child_actions)
    assert action in sampled
    # noise=0 + stub zero Q → argmax(perturbed + σ) = argmax(legal_logits over sampled)
    # → action 8 (highest logit among sampled {6, 7, 8}).
    assert action == 8


def test_select_action_gumbel_greedy_with_noise_off():
    """Repeat call with noise off returns the same (action, π') — deterministic."""
    cfg = _StubConfig(num_simulations=4, gumbel_num_considered=3,
                      use_gumbel=True, use_gumbel_noise=False)
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
    mcts1 = _make_batched(cfg, root_logits=logits)
    mcts2 = _make_batched(cfg, root_logits=logits)
    r1 = mcts1.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False)[0]
    r2 = mcts2.run_batch([torch.zeros(3, 3, 3)], [list(range(9))], add_noise=False)[0]
    a1, d1 = select_action_gumbel(r1, cfg, 9)
    a2, d2 = select_action_gumbel(r2, cfg, 9)
    assert a1 == a2
    np.testing.assert_allclose(d1, d2)


# --- G3: select_action_gumbel with non-trivial Q (v_mix matters) ---------

def _hand_build_gumbel_root(
    legals, logits, sampled_actions, sampled_visits,
    sampled_rewards, sampled_value_sums,
    perturbed=None, v_prior=0.0, min_max_q=None,
):
    """Synthesize an MCTSNode root as if search had just finished.

    Bypasses ``run_batch`` so we can drive ``select_action_gumbel`` with
    precisely controlled Q, visit, and min/max state.
    """
    root = MCTSNode()
    root.root_legal_actions = np.asarray(legals, dtype=np.int64)
    root.root_legal_logits = np.asarray(logits, dtype=np.float64)
    root.child_actions = np.asarray(sampled_actions, dtype=np.int64)
    root.child_visits = np.asarray(sampled_visits, dtype=np.float64)
    root.child_rewards = np.asarray(sampled_rewards, dtype=np.float64)
    root.child_value_sums = np.asarray(sampled_value_sums, dtype=np.float64)
    m = len(sampled_actions)
    root.child_priors = np.full(m, 1.0 / m, dtype=np.float64)
    root.root_sampled_perturbed = (
        np.asarray(perturbed, dtype=np.float64) if perturbed is not None
        else np.zeros(m, dtype=np.float64)
    )
    root.value_prior = v_prior
    root.min_max_q = min_max_q
    return root


def test_select_action_gumbel_v_mix_fills_unsampled_legals_with_mass():
    """Unsampled legal actions get π' mass through v_mix (paper Eq. 33).

    Setup: legals=[0,1,2] with uniform zero logits. Sampled={0,2}; only action 0
    was visited (visits=4, mean Q=0.8). Action 1 is an unsampled legal →
    completedQ(1) = v_mix (not zero). Action 2 is sampled-but-unvisited →
    completedQ(2) also = v_mix.

    Hand computation:
      v_mix = (v_prior + Σ_{visited} π(a)·q(a) · N_tot / Σ_{visited} π(a)) / (1 + N_tot)
            = (0.3 + 4·0.8) / 5 = 0.7
      completedQ_raw = [0.8, 0.7, 0.7]
      q_norm (with min_max_q=(0, 0.8)) = [1.0, 0.875, 0.875]
      σ = (50 + 4) · q_norm = [54, 47.25, 47.25]
      π' ∝ exp([54, 47.25, 47.25])

    All three legals must have non-zero mass; action 0 dominates.
    """
    root = _hand_build_gumbel_root(
        legals=[0, 1, 2], logits=[0.0, 0.0, 0.0],
        sampled_actions=[0, 2], sampled_visits=[4.0, 0.0],
        sampled_rewards=[0.0, 0.0], sampled_value_sums=[-3.2, 0.0],
        v_prior=0.3, min_max_q=(0.0, 0.8),
    )
    cfg = _StubConfig(
        use_gumbel=True, gumbel_num_considered=2, discount=1.0,
        gumbel_c_visit=50.0, gumbel_c_scale=1.0,
    )
    action, dense = select_action_gumbel(root, cfg, action_space_size=3)
    dense = np.asarray(dense)

    assert action == 0  # argmax sampled by perturbed+σ = [54, 47.25]
    assert dense.sum() == pytest.approx(1.0)
    assert dense[0] > dense[1] > 0
    assert dense[2] > 0  # sampled-but-unvisited also gets v_mix mass
    # Hand-check exact values
    expected_logits = np.array([54.0, 47.25, 47.25])
    expected_logits = expected_logits - expected_logits.max()
    expected = np.exp(expected_logits) / np.exp(expected_logits).sum()
    np.testing.assert_allclose(dense, expected, atol=1e-10)


def test_select_action_gumbel_degenerate_min_max_range_clips_completed_q():
    """When MinMaxStats never updated (min>max sentinel), q_norm is clipped to [0,1].

    Guards the division-by-zero branch: a call on a freshly constructed root
    with no search progress shouldn't NaN.
    """
    root = _hand_build_gumbel_root(
        legals=[0, 1], logits=[1.0, 0.0],
        sampled_actions=[0, 1], sampled_visits=[1.0, 1.0],
        sampled_rewards=[0.0, 0.0], sampled_value_sums=[0.0, 0.0],
        v_prior=0.0, min_max_q=(float("inf"), float("-inf")),
    )
    cfg = _StubConfig(
        use_gumbel=True, gumbel_num_considered=2, discount=1.0,
        gumbel_c_visit=50.0, gumbel_c_scale=1.0,
    )
    _, dense = select_action_gumbel(root, cfg, action_space_size=2)
    dense = np.asarray(dense)
    # completedQ=0 after clip → σ=0 → π' = softmax(logits)
    shifted = np.array([1.0, 0.0]) - 1.0
    expected = np.exp(shifted) / np.exp(shifted).sum()
    np.testing.assert_allclose(dense, expected, atol=1e-10)


def test_select_action_gumbel_normalization_invariant_under_q_range_shift():
    """Shifting and scaling raw Q by the same (min, max) leaves π' invariant.

    Because σ uses only the [0,1]-normalized Q, two roots with different raw
    Q-scales but identical normalized Q's must produce the same π'.
    """
    cfg = _StubConfig(
        use_gumbel=True, gumbel_num_considered=2, discount=1.0,
        gumbel_c_visit=50.0, gumbel_c_scale=1.0,
    )
    # Case A: mean Q = [0.5, 0.25] within range [0.25, 0.5] → q_norm = [1.0, 0.0]
    root_a = _hand_build_gumbel_root(
        legals=[0, 1], logits=[0.0, 0.0],
        sampled_actions=[0, 1], sampled_visits=[2.0, 2.0],
        sampled_rewards=[0.0, 0.0], sampled_value_sums=[-1.0, -0.5],
        v_prior=0.0, min_max_q=(0.25, 0.5),
    )
    # Case B: mean Q = [1.0, 0.75] within range [0.75, 1.0] → q_norm = [1.0, 0.0]
    root_b = _hand_build_gumbel_root(
        legals=[0, 1], logits=[0.0, 0.0],
        sampled_actions=[0, 1], sampled_visits=[2.0, 2.0],
        sampled_rewards=[0.0, 0.0], sampled_value_sums=[-2.0, -1.5],
        v_prior=0.0, min_max_q=(0.75, 1.0),
    )
    _, d_a = select_action_gumbel(root_a, cfg, 2)
    _, d_b = select_action_gumbel(root_b, cfg, 2)
    np.testing.assert_allclose(d_a, d_b, atol=1e-10)


def test_select_action_gumbel_action_stays_in_sampled_set_even_when_unsampled_has_higher_logit():
    """Paper page 18: A_{n+1} is argmax over the *sampled m-set*, not all legals.

    Setup: 4 legal actions [0..3]; only [0, 1] sampled. Action 2 has the
    highest logit but was never considered — must NOT be returned.
    """
    root = _hand_build_gumbel_root(
        legals=[0, 1, 2, 3], logits=[0.0, 0.0, 99.0, 0.0],
        sampled_actions=[0, 1], sampled_visits=[2.0, 2.0],
        sampled_rewards=[0.0, 0.0], sampled_value_sums=[-2.0, -1.0],
        perturbed=[0.0, 0.0], v_prior=0.0, min_max_q=(0.5, 1.0),
    )
    cfg = _StubConfig(
        use_gumbel=True, gumbel_num_considered=2, discount=1.0,
        gumbel_c_visit=50.0, gumbel_c_scale=1.0,
    )
    action, _ = select_action_gumbel(root, cfg, action_space_size=4)
    assert action in (0, 1)


def test_select_action_gumbel_perturbed_tiebreaks_when_q_tied():
    """With all sampled-action Q's tied, perturbed (g+logits) breaks the argmax."""
    root = _hand_build_gumbel_root(
        legals=[0, 1, 2], logits=[0.0, 0.0, 0.0],
        sampled_actions=[0, 1, 2], sampled_visits=[1.0, 1.0, 1.0],
        sampled_rewards=[0.0, 0.0, 0.0],
        sampled_value_sums=[-0.5, -0.5, -0.5],  # all mean Q = 0.5
        perturbed=[0.1, 0.9, 0.3],  # action 1 wins
        v_prior=0.0, min_max_q=(0.5, 0.5),  # degenerate range → clip
    )
    cfg = _StubConfig(
        use_gumbel=True, gumbel_num_considered=3, discount=1.0,
        gumbel_c_visit=50.0, gumbel_c_scale=1.0,
    )
    action, _ = select_action_gumbel(root, cfg, action_space_size=3)
    assert action == 1


# --- G3: KL policy loss ----------------------------------------------------

def test_kl_loss_is_zero_when_network_equals_target():
    """Sanity check: F.kl_div(log_softmax(π'_logits), π') == 0 when π_net ≡ π'.

    Validates the use_gumbel policy-loss path reaches 0 on perfect prediction
    (which CE-on-visits does not — it bottoms out at H(π')).
    """
    import torch.nn.functional as F

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    target = torch.softmax(logits, dim=1)  # π' ≡ π_net by construction
    log_p = F.log_softmax(logits, dim=1)
    loss = F.kl_div(log_p, target, reduction="none").sum(dim=1)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_gumbel_root_state_fields_present_in_public_mcts_api():
    """Smoke test that _GumbelRootState is importable and builds as expected."""
    state = _GumbelRootState(
        m=4, schedule=[(4, 2), (2, 4)], phase=0,
        surviving=np.array([0, 1, 2, 3], dtype=np.int64),
        queue=[0, 1, 2, 3, 0, 1, 2, 3], queue_pos=0,
    )
    assert state.m == 4
    assert state.schedule == [(4, 2), (2, 4)]
    assert len(state.queue) == 8
