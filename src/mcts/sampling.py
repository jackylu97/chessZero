"""Sampling utilities for Sampled MuZero.

Gumbel-Top-K (Kool et al. 2019, "Stochastic Beams and Where to Find Them")
samples K distinct indices without replacement from the categorical distribution
softmax(logits). For K=1 it reduces to the standard Gumbel-max trick.
"""

from __future__ import annotations

import numpy as np
import torch


def sample_topk_gumbel(
    logits,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample K distinct indices from softmax(logits) without replacement.

    Args:
        logits: (A,) or (B, A) torch tensor or numpy array.
        k: number of indices to sample. If k >= A, returns every index.
        rng: numpy Generator controlling the Gumbel noise.

    Returns:
        actions: np.int64 array of shape (K,) or (B, K).
        perturbed: same shape — logits + Gumbel noise at the sampled indices
                   (useful for diagnostics; returned in descending order).
    """
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy().astype(np.float64)
    else:
        logits_np = np.asarray(logits, dtype=np.float64)

    u = rng.uniform(size=logits_np.shape)
    np.clip(u, 1e-20, 1.0 - 1e-20, out=u)
    gumbel = -np.log(-np.log(u))
    perturbed = logits_np + gumbel

    a = perturbed.shape[-1]
    k_eff = min(k, a)

    if perturbed.ndim == 1:
        if k_eff == a:
            idx = np.arange(a, dtype=np.int64)
        else:
            idx = np.argpartition(-perturbed, kth=k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-perturbed[idx])]
        return idx.astype(np.int64), perturbed[idx]

    b = perturbed.shape[0]
    if k_eff == a:
        idx = np.broadcast_to(np.arange(a), perturbed.shape).copy()
    else:
        idx = np.argpartition(-perturbed, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(b)[:, None]
    order = np.argsort(-perturbed[rows, idx], axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    return idx.astype(np.int64), np.take_along_axis(perturbed, idx, axis=1)
