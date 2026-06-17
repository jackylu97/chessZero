"""Representation-informativeness metrics.

Shared by ``scripts/probe_representation.py`` (offline, across checkpoints) and
``MuZeroTrainer`` (in-loop TensorBoard logging). All functions operate on a
latent matrix ``H`` of shape [N, D] (one flattened representation per position)
and optional scalar labels.

The headline metric is ``r2_eval``: the validation R2 of a ridge map H -> label,
i.e. how much of the label is LINEARLY decodable from the frozen representation
(Alain & Bengio linear-probe methodology). A representation that is collapsing
into the draw basin shows ``crosspos_cos`` -> 1, ``eff_rank`` shrinking, and
``r2_eval`` falling even while the value LOSS looks healthy.
"""

import numpy as np


def effective_rank(H):
    """Entropy effective rank and participation ratio of the latent covariance."""
    Hc = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    s2 = s ** 2
    if s2.sum() <= 0:
        return 0.0, 0.0
    p = s2 / s2.sum()
    p = p[p > 0]
    ent_rank = float(np.exp(-(p * np.log(p)).sum()))          # exp(spectral entropy)
    part_ratio = float((s2.sum() ** 2) / (s2 ** 2).sum())      # participation ratio
    return ent_rank, part_ratio


def mean_crosspos_cosine(H, max_pairs=200000, seed=0):
    """Mean cosine over ALL distinct position pairs (exact closed form). ->1.0 = collapse.

    For unit vectors u_i, mean_{i!=j} <u_i,u_j> = (||sum_i u_i||^2 - sum_i||u_i||^2)
    / (n(n-1)). This is O(n*D) and exact, replacing the old O(max_pairs*D) random
    sampling — which both approximated the mean AND materialized multi-GB gather
    temporaries (norm[i], norm[j], their product), the transient that OOM-killed the
    repr probe at step boundaries. ``max_pairs``/``seed`` are accepted for backward
    compatibility and ignored.
    """
    n = len(H)
    if n < 2:
        return float("nan")
    norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    s = norm.sum(axis=0)
    total = float(s @ s)               # sum_{i,j} <u_i,u_j> = ||sum_i u_i||^2
    diag = float((norm * norm).sum())  # sum_i <u_i,u_i>
    return (total - diag) / (n * (n - 1))


def dual_ridge_r2(X, y, lam_grid=(1e1, 1e2, 1e3, 1e4), val_frac=0.3, seed=0):
    """Best-of-grid validation R2 of a ridge map X->y, dual form (handles D>N).

    Returns (best_r2, (val_pred, val_y)). NaN labels are dropped first.
    """
    finite = np.isfinite(y)
    X, y = X[finite], y[finite]
    if len(y) < 20 or float(np.std(y)) < 1e-6:
        return float("nan"), (np.zeros(0), np.zeros(0))
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nval = max(1, int(n * val_frac))
    vi, ti = idx[:nval], idx[nval:]
    Xt, yt, Xv, yv = Xs[ti], y[ti], Xs[vi], y[vi]
    yt_m = float(yt.mean())
    Ktt = Xt @ Xt.T
    Ktv = Xt @ Xv.T
    eye = np.eye(len(yt))
    ss_tot = ((yv - yv.mean()) ** 2).sum() + 1e-12
    best_r2, best_pred = -np.inf, (np.zeros(0), np.zeros(0))
    for lam in lam_grid:
        alpha = np.linalg.solve(Ktt + lam * eye, (yt - yt_m))
        pred = Ktv.T @ alpha + yt_m
        r2 = 1.0 - ((yv - pred) ** 2).sum() / ss_tot
        if r2 > best_r2:
            best_r2, best_pred = float(r2), (pred, yv)
    return best_r2, best_pred


def compute_repr_metrics(H, ev, out, seed=0):
    """Full metric dict from latents H [N,D] and scalar labels ev, out [N].

    ev  = Stockfish STM eval (may be all-NaN if unavailable -> r2_eval NaN).
    out = STM game outcome in {-1,0,+1}.
    """
    H = np.asarray(H, dtype=np.float64)
    ev = np.asarray(ev, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    cpc = mean_crosspos_cosine(H, seed=seed)
    ent_rank, part_ratio = effective_rank(H)
    r2_eval, _ = dual_ridge_r2(H, ev, seed=seed)
    r2_out, op = dual_ridge_r2(H, out, seed=seed)
    pred, yv = op
    dec = np.abs(yv) > 0.5
    sign_acc = float((np.sign(pred[dec]) == np.sign(yv[dec])).mean()) if dec.sum() else float("nan")
    return {
        "crosspos_cos": cpc,
        "eff_rank": ent_rank,
        "part_ratio": part_ratio,
        "r2_eval": r2_eval,
        "r2_out": r2_out,
        "sign_acc": sign_acc,
        "n": int(len(H)),
    }
