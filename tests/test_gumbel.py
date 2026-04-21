"""Unit tests for src/mcts/gumbel.py — paper Eqs 8, 10, 11, 33 + Figure 1."""

import math

import numpy as np
import pytest

from src.mcts.gumbel import (
    compute_completed_q,
    compute_improved_policy,
    compute_v_mix,
    sequential_halving_schedule,
    sigma,
)


# --- sigma -----------------------------------------------------------------

def test_sigma_paper_value():
    # Paper Eq. 8 with c_visit=50, c_scale=1, max_n=100, q=0.5 → (50+100)*1*0.5 = 75
    assert sigma(np.array([0.5]), max_n=100)[0] == pytest.approx(75.0)


def test_sigma_monotonic_in_q():
    q = np.array([0.1, 0.5, 0.9])
    s = sigma(q, max_n=10)
    assert np.all(np.diff(s) > 0)


def test_sigma_scales_with_max_n():
    q = np.array([0.7])
    small = sigma(q, max_n=0)[0]
    large = sigma(q, max_n=1000)[0]
    assert large > small
    assert large / small == pytest.approx(1050.0 / 50.0)


def test_sigma_zero_q_gives_zero():
    assert sigma(np.zeros(5), max_n=999)[0] == 0.0


# --- compute_v_mix ---------------------------------------------------------

def test_v_mix_no_visits_returns_prior():
    v = compute_v_mix(v_prior=0.42, priors=np.array([0.3, 0.3, 0.4]),
                     visits=np.array([0, 0, 0]), q_values=np.array([0.0, 0.0, 0.0]))
    assert v == pytest.approx(0.42)


def test_v_mix_all_visited_hand_computed():
    # priors uniform 1/3, visits=[1,1,1], q=[0.1, 0.2, 0.3], v_prior=0.5
    # visited_prior_sum = 1, weighted_q = (1/3*(0.1+0.2+0.3))/1 = 0.2
    # v_mix = (0.5 + 3*0.2) / 4 = 1.1/4 = 0.275
    v = compute_v_mix(
        v_prior=0.5,
        priors=np.array([1/3, 1/3, 1/3]),
        visits=np.array([1, 1, 1]),
        q_values=np.array([0.1, 0.2, 0.3]),
    )
    assert v == pytest.approx(0.275)


def test_v_mix_one_visited_hand_computed():
    # priors=[0.2, 0.5, 0.3], visits=[0, 3, 0], q=[_, 0.8, _], v_prior=0.1
    # total_visits=3, visited_prior_sum=0.5, weighted_q = (0.5*0.8)/0.5 = 0.8
    # v_mix = (0.1 + 3*0.8) / (1+3) = 2.5/4 = 0.625
    v = compute_v_mix(
        v_prior=0.1,
        priors=np.array([0.2, 0.5, 0.3]),
        visits=np.array([0, 3, 0]),
        q_values=np.array([0.0, 0.8, 0.0]),
    )
    assert v == pytest.approx(0.625)


def test_v_mix_visited_prior_sum_zero_returns_prior():
    # degenerate: visited action has zero prior
    v = compute_v_mix(
        v_prior=0.3,
        priors=np.array([0.0, 1.0]),
        visits=np.array([5, 0]),
        q_values=np.array([0.9, 0.0]),
    )
    assert v == pytest.approx(0.3)


# --- compute_completed_q ---------------------------------------------------

def test_completed_q_visited_unvisited_mix():
    visits = np.array([2, 0, 3, 0])
    q = np.array([0.7, 0.0, 0.4, 0.0])
    out = compute_completed_q(visits, q, v_mix=0.25)
    assert np.allclose(out, [0.7, 0.25, 0.4, 0.25])


def test_completed_q_none_visited_all_vmix():
    out = compute_completed_q(
        visits=np.zeros(4),
        q_values=np.zeros(4),
        v_mix=0.55,
    )
    assert np.allclose(out, np.full(4, 0.55))


# --- compute_improved_policy -----------------------------------------------

def test_improved_policy_sums_to_one():
    logits = np.array([0.1, 2.0, -1.0, 0.5])
    q = np.array([0.5, 0.5, 0.5, 0.5])
    p = compute_improved_policy(logits, q, max_n=10)
    assert p.sum() == pytest.approx(1.0)


def test_improved_policy_equal_q_reduces_to_softmax_logits():
    # σ(constant q) adds the same scalar to every logit → softmax unchanged
    logits = np.array([0.1, 2.0, -1.0, 0.5])
    q = np.full(4, 0.5)
    p = compute_improved_policy(logits, q, max_n=50)

    shifted = logits - logits.max()
    expected = np.exp(shifted) / np.exp(shifted).sum()
    assert np.allclose(p, expected)


def test_improved_policy_higher_q_gets_higher_mass():
    logits = np.array([0.0, 0.0, 0.0])  # uniform prior
    q = np.array([0.1, 0.5, 0.9])       # a=2 is best
    p = compute_improved_policy(logits, q, max_n=20)
    assert p[0] < p[1] < p[2]


def test_improved_policy_neg_inf_logit_yields_zero_mass():
    logits = np.array([0.0, float("-inf"), 1.0])
    q = np.array([0.5, 0.9, 0.5])  # q for the masked action is high but irrelevant
    p = compute_improved_policy(logits, q, max_n=10)
    assert p[1] == 0.0
    assert p[0] + p[2] == pytest.approx(1.0)


# --- sequential_halving_schedule -------------------------------------------

def test_schedule_paper_figure1_n200_m16():
    # Paper Figure 1: n=200, m=16 → 4 phases
    # Per-phase: (m_p, max(1, n // (log2(m) · m_p))) with log2(16)=4
    #   (16, 200//64=3), (8, 200//32=6), (4, 200//16=12), (2, 200//8=25)
    sched = sequential_halving_schedule(num_sims=200, m=16)
    assert sched == [(16, 3), (8, 6), (4, 12), (2, 25)]


def test_schedule_m_equals_one_single_phase():
    sched = sequential_halving_schedule(num_sims=50, m=1)
    assert sched == [(1, 50)]


def test_schedule_m_equals_n_first_phase_one_visit():
    # m=16, n=16: log2(16)=4, phase 0 visits = max(1, 16//64) = 1, total phase 0 = 16.
    # Budget exhausts; schedule has only phase 0.
    sched = sequential_halving_schedule(num_sims=16, m=16)
    assert sched[0] == (16, 1)
    assert sum(m_p * v for m_p, v in sched) <= 16


def test_schedule_total_does_not_exceed_budget():
    for n, m in [(100, 16), (50, 8), (25, 4), (200, 32), (8, 8)]:
        sched = sequential_halving_schedule(num_sims=n, m=m)
        total = sum(m_p * v for m_p, v in sched)
        assert total <= n, f"schedule exceeds budget for n={n}, m={m}: {sched} (total={total})"


def test_schedule_phase_count_is_log2_m():
    sched = sequential_halving_schedule(num_sims=200, m=16)
    assert len(sched) == int(math.log2(16))


def test_schedule_m_p_halves_each_phase():
    sched = sequential_halving_schedule(num_sims=400, m=32)
    m_ps = [s[0] for s in sched]
    # Expect 32, 16, 8, 4, 2 — halving each phase
    for i in range(1, len(m_ps)):
        assert m_ps[i] == m_ps[i-1] // 2
