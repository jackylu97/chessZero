"""Tests for decisive-game resampling (decisive_sample_frac).

Oversamples |z|=1 (decisive) self-play games into every batch so a rising
self-play draw rate can't wash the value signal out (the draw-saturation loop).
Mirrors the warmstart stratification; is_warmstart stays keyed on external_values.
"""
import numpy as np
import torch

from src.training.replay_buffer import GameHistory, ReplayBuffer


def _game(outcome, length=6, action_space=9):
    g = GameHistory()
    for i in range(length):
        g.observations.append(torch.full((3, 3, 3), float(i)))
        g.actions.append(i % action_space)
        g.rewards.append(0.0)
        g.policies.append(np.full(action_space, 1 / action_space, dtype=np.float32))
        g.root_values.append(0.0)
    g.game_outcome = float(outcome)
    g.legal_actions_list = [list(range(action_space)) for _ in range(length)]
    return g


def _buffer(n_decisive, n_draw):
    buf = ReplayBuffer(max_size=1000)
    for _ in range(n_decisive):
        buf.save_game(_game(outcome=np.random.choice([-1.0, 1.0])))
    for _ in range(n_draw):
        buf.save_game(_game(outcome=0.0))
    return buf


def _decisive_count(buf, game_indices):
    return sum(1 for i in game_indices if abs(buf.buffer[i].game_outcome) >= 0.5)


def test_decisive_frac_forces_decisive_fraction():
    """With decisive_sample_frac=0.5, exactly round(B*0.5) sampled games are decisive,
    far above the natural 4/20 = 0.2 rate."""
    np.random.seed(0)
    buf = _buffer(n_decisive=4, n_draw=16)  # natural decisive frac = 0.2
    B = 32
    _, game_indices, weights = buf.sample_batch(
        B, num_unroll_steps=3, td_steps=-1, discount=1.0, action_space_size=9,
        alpha=0.6, beta=1.0, value_head_type="wdl", decisive_sample_frac=0.5,
    )
    n_dec = _decisive_count(buf, game_indices)
    assert n_dec == round(B * 0.5), f"expected {round(B*0.5)} decisive, got {n_dec}"
    assert np.isfinite(weights).all() and len(weights) == B


def test_decisive_frac_zero_is_flat():
    """decisive_sample_frac=0.0 → no forced decisive fraction (≈ natural rate)."""
    np.random.seed(0)
    buf = _buffer(n_decisive=4, n_draw=46)  # natural decisive frac = 0.08
    B = 64
    fracs = []
    for _ in range(8):
        _, gi, _ = buf.sample_batch(
            B, 3, -1, 1.0, 9, alpha=0.6, beta=1.0, value_head_type="wdl",
            decisive_sample_frac=0.0,
        )
        fracs.append(_decisive_count(buf, gi) / B)
    # Flat sampling should sit near the natural 0.08, nowhere near a forced 0.5.
    assert np.mean(fracs) < 0.3


def test_decisive_frac_no_decisive_games_falls_back():
    """No decisive games in the buffer → falls back to flat sampling, no crash."""
    np.random.seed(0)
    buf = _buffer(n_decisive=0, n_draw=20)  # all draws
    B = 16
    _, gi, w = buf.sample_batch(
        B, 3, -1, 1.0, 9, alpha=0.6, beta=1.0, value_head_type="wdl",
        decisive_sample_frac=0.5,
    )
    assert len(gi) == B and np.isfinite(w).all()
    assert _decisive_count(buf, gi) == 0  # nothing decisive to sample


def test_is_warmstart_unaffected_by_decisive_resampling():
    """is_warmstart keys on external_values, not decisiveness — stays all-False
    for self-play-only buffers even under decisive resampling."""
    np.random.seed(0)
    buf = _buffer(n_decisive=8, n_draw=8)
    batch, _, _ = buf.sample_batch(
        16, 3, -1, 1.0, 9, alpha=0.6, beta=1.0, value_head_type="wdl",
        decisive_sample_frac=0.5,
    )
    assert not batch["is_warmstart"].any()
