"""Legal-move policy masking (config.mask_illegal_policy).

When enabled, the policy loss keeps the standard full-softmax cross-entropy (which
learns legality for free via the shared normalizer) and ADDS the full-softmax
probability mass on illegal moves as a separate penalty signal driving it below
the CE's natural floor. Disabled, it is exactly the reference CE with zero penalty.
We do NOT renormalize the softmax over legal moves — that is shift-invariant over
the legal logits and cannot teach legality: full_CE = masked_CE - log P(legal).

`_policy_terms` does not use `self`, so we call it unbound with self=None.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.training.trainer import MuZeroTrainer
from src.training.replay_buffer import GameHistory, ReplayBuffer

PT = MuZeroTrainer._policy_terms
# In the trainer, policy_loss_fn is the BOUND self._policy_loss; emulate that
# binding here (these methods don't actually use self).
PL = lambda _self, logits, targets: MuZeroTrainer._policy_loss(None, logits, targets)
_BASE = lambda logits, targets: MuZeroTrainer._policy_loss(None, logits, targets)


def _legal_mask(A, legal):
    m = torch.zeros(A)
    m[legal] = 1.0
    return m


def test_mask_none_reproduces_reference_ce_and_zero_penalty():
    torch.manual_seed(0)
    A = 20
    logits = torch.randn(4, A)
    legal = [1, 5, 9]
    targets = torch.zeros(4, A)
    targets[:, legal] = torch.softmax(torch.randn(4, len(legal)), dim=1)
    ce, pen = PT(None, logits, targets, None, _BASE)
    ref = PL(None, logits, targets)
    torch.testing.assert_close(ce, ref)
    torch.testing.assert_close(pen, torch.zeros(4))


def test_masking_uses_full_softmax_ce_plus_illegal_penalty():
    """With a mask, the CE is the SAME standard full-softmax CE (NOT renormalized
    over legal); the mask only produces the illegal-mass penalty."""
    torch.manual_seed(1)
    A = 30
    logits = torch.randn(3, A)
    legal = [2, 7, 11, 19]
    mask = torch.stack([_legal_mask(A, legal)] * 3)
    targets = torch.zeros(3, A)
    targets[:, legal] = torch.softmax(torch.randn(3, len(legal)), dim=1)

    ce, pen = PT(None, logits, targets, mask, _BASE)

    # CE == the plain full-softmax CE (masking does NOT change it).
    torch.testing.assert_close(ce, _BASE(logits, targets), atol=1e-6, rtol=1e-4)
    # Penalty == full-softmax probability mass on illegal moves (positive).
    full = F.softmax(logits, dim=1)
    expected_pen = full.sum(dim=1) - full[:, legal].sum(dim=1)
    torch.testing.assert_close(pen, expected_pen, atol=1e-6, rtol=1e-4)
    assert (pen > 0).all()


def test_full_ce_equals_masked_ce_minus_log_p_legal():
    """The decomposition that motivates (a): full_CE = masked_CE - log P(legal).
    Renormalizing the CE over legal moves drops the -log P(legal) legality term,
    which is exactly the signal that teaches the head to suppress illegal moves."""
    torch.manual_seed(3)
    A = 24
    logits = torch.randn(5, A)
    legal = [1, 4, 9, 15, 20]
    targets = torch.zeros(5, A)
    targets[:, legal] = torch.softmax(torch.randn(5, len(legal)), dim=1)

    full_ce = _BASE(logits, targets)
    masked_ce = -(targets[:, legal] * F.log_softmax(logits[:, legal], dim=1)).sum(dim=1)
    log_p_legal = torch.log(F.softmax(logits, dim=1)[:, legal].sum(dim=1))
    torch.testing.assert_close(full_ce, masked_ce - log_p_legal, atol=1e-5, rtol=1e-4)


def test_no_nan_with_extreme_logits():
    A = 25
    logits = torch.full((2, A), -50.0)
    legal = [4, 12]
    logits[:, legal] = torch.tensor([2.0, 1.0])
    mask = torch.stack([_legal_mask(A, legal)] * 2)
    targets = torch.zeros(2, A)
    targets[:, legal] = 0.5
    ce, pen = PT(None, logits, targets, mask, _BASE)
    assert torch.isfinite(ce).all()
    assert torch.isfinite(pen).all()


def _toy_game(n=6, A=20):
    g = GameHistory(game_name="tictactoe")
    g.game_outcome = 1.0
    rng = np.random.default_rng(0)
    for i in range(n):
        g.observations.append(torch.zeros(1, 3, 3))
        g.actions.append(i % A)
        g.policies.append(np.full(A, 1.0 / A, dtype=np.float32))
        g.root_values.append(0.0)
        # distinct legal sets per ply so we can verify alignment
        g.legal_actions_list.append(sorted(rng.choice(A, size=3, replace=False).tolist()))
        g.rewards.append(0.0)
    g.observations.append(torch.zeros(1, 3, 3))
    return g


def test_sample_batch_builds_aligned_legal_masks():
    A = 20
    buf = ReplayBuffer(max_size=100)
    g = _toy_game(n=6, A=A)
    buf.save_game(g)
    K = 3
    batch, _, _ = buf.sample_batch(
        batch_size=8, num_unroll_steps=K, td_steps=5, discount=0.99,
        action_space_size=A, value_head_type="wdl", history_frames=1,
        build_legal_masks=True,
    )
    assert "target_legal_masks" in batch
    lm = batch["target_legal_masks"]
    assert lm.shape == (8, K + 1, A)
    # Every row is either all-ones (past-end fallback) or a {0,1} mask whose
    # support matches some recorded legal set.
    assert set(torch.unique(lm).tolist()) <= {0.0, 1.0}
    legal_sets = {tuple(l) for l in g.legal_actions_list}
    for b in range(8):
        for k in range(K + 1):
            row = lm[b, k]
            if row.sum() == A:
                continue  # all-ones fallback
            support = tuple(torch.nonzero(row, as_tuple=True)[0].tolist())
            assert support in legal_sets


def test_disabled_omits_legal_masks():
    A = 20
    buf = ReplayBuffer(max_size=100)
    buf.save_game(_toy_game(n=6, A=A))
    batch, _, _ = buf.sample_batch(
        batch_size=4, num_unroll_steps=3, td_steps=5, discount=0.99,
        action_space_size=A, value_head_type="wdl", history_frames=1,
        build_legal_masks=False,
    )
    assert "target_legal_masks" not in batch
