"""Gumbel MuZero utilities (Danihelka et al. 2022).

Implements the math pieces needed for Plain Gumbel MuZero:
    - sigma transform (paper Eq. 8)
    - v_mix approximation for unvisited completed Q-values (paper Eq. 33 / Appendix D)
    - completed Q-values (paper Eq. 10)
    - improved policy π' = softmax(logits + σ(completedQ)) (paper Eq. 11)
    - Sequential Halving schedule (paper Figure 1 + Appendix F)

All functions are pure numpy. No torch, no I/O, no tree state.

Cross-referenced against LightZero ``lzero/mcts/ctree/ctree_gumbel_muzero/lib/cnode.cpp``
(``compute_mixed_value``, ``qtransform_completed_by_mix_value``,
``get_sequence_of_considered_visits``).
"""

from __future__ import annotations

import math

import numpy as np


def sigma(q_norm: np.ndarray, max_n: float,
          c_visit: float = 50.0, c_scale: float = 1.0) -> np.ndarray:
    """Monotonic Q→logit-increment transform (paper Eq. 8).

    σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂

    Q-values must be pre-normalized to [0, 1] (paper F p. 17 — done via MinMaxStats
    in our codebase). Paper hyperparams: c_visit=50, c_scale=1.0 — same across all
    sim budgets on Go and chess.
    """
    return (c_visit + float(max_n)) * c_scale * np.asarray(q_norm, dtype=np.float64)


def compute_v_mix(v_prior: float, priors: np.ndarray,
                  visits: np.ndarray, q_values: np.ndarray) -> float:
    """Mixed-value approximation of v_π (paper Eq. 33, Appendix D).

        v_mix = (1 / (1 + ΣN(b))) · [ v̂_π + (ΣN(b) / Σ_{b: N(b)>0} π(b))
                                             · Σ_{a: N(a)>0} π(a) q(a) ]

    Interpolates the value-network prior v̂_π with the prior-weighted mean of the
    Q-values observed so far. Used to fill in completedQ(a) for unvisited actions.

    Edge cases:
      - no visits: returns v_prior
      - all visited priors sum to 0 (degenerate): returns v_prior
    """
    priors = np.asarray(priors, dtype=np.float64)
    visits = np.asarray(visits, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)

    total_visits = float(visits.sum())
    if total_visits <= 0.0:
        return float(v_prior)

    visited = visits > 0
    visited_prior_sum = float(priors[visited].sum())
    if visited_prior_sum <= 0.0:
        return float(v_prior)

    weighted_q = float((priors[visited] * q_values[visited]).sum()) / visited_prior_sum
    return (float(v_prior) + total_visits * weighted_q) / (1.0 + total_visits)


def compute_completed_q(visits: np.ndarray, q_values: np.ndarray,
                        v_mix: float) -> np.ndarray:
    """completedQ(a) = q(a) if N(a) > 0 else v_mix (paper Eq. 10)."""
    visits = np.asarray(visits, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    return np.where(visits > 0, q_values, float(v_mix))


def compute_improved_policy(logits: np.ndarray, completed_q_norm: np.ndarray,
                            max_n: float,
                            c_visit: float = 50.0,
                            c_scale: float = 1.0) -> np.ndarray:
    """Improved policy π' = softmax(logits + σ(completedQ)) (paper Eq. 11).

    ``completed_q_norm`` must be normalized to [0, 1]. ``logits`` should be the
    *raw* policy-network logits (pre-softmax); for unsampled / illegal actions,
    pass -inf so softmax assigns them zero mass.
    """
    logits = np.asarray(logits, dtype=np.float64)
    sig = sigma(completed_q_norm, max_n, c_visit, c_scale)
    x = logits + sig
    x = x - x.max()
    e = np.exp(x)
    total = e.sum()
    if total <= 0.0:
        out = np.zeros_like(e)
        return out
    return e / total


def sequential_halving_schedule(num_sims: int, m: int) -> list[tuple[int, int]]:
    """Paper Figure 1 / F p. 18 — per-phase (m_p, visits_per_action_p).

    Per-phase budget is n / ⌈log2(m)⌉; each surviving action gets
    ``max(1, ⌊n / (⌈log2(m)⌉ · m_p)⌋)`` visits. After each phase the action set
    is halved by dropping the bottom-ranked half under
    g(a) + logits(a) + σ(q̂(a)).

    Terminates when the remaining simulation budget is exhausted or only one
    action remains.

    With m=1 (single action), returns a single phase absorbing all sims. With
    m > num_sims, phase 0 visits every action once and the budget runs out
    before any halving happens (caller should handle this).
    """
    if m <= 1:
        return [(1, max(1, int(num_sims)))]

    log2m = max(1, math.ceil(math.log2(m)))
    schedule: list[tuple[int, int]] = []
    m_p = int(m)
    budget_left = int(num_sims)

    for _ in range(log2m):
        if m_p < 1 or budget_left <= 0:
            break
        visits_per_action = max(1, num_sims // (log2m * m_p))
        phase_total = visits_per_action * m_p
        if phase_total > budget_left:
            # Use remaining budget; visit as many as we can
            visits_per_action = max(1, budget_left // m_p)
            phase_total = visits_per_action * m_p
        schedule.append((m_p, visits_per_action))
        budget_left -= phase_total
        if m_p == 1:
            break
        m_p = max(1, m_p // 2)

    return schedule
