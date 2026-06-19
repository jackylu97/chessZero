"""Triton kernels for ``TensorMCTS``.

Currently provides a fused PUCT-step kernel that replaces ~25 individual
PyTorch ops per inner-loop iteration of ``_select`` with a single Triton
launch. One program per game; each program reads a row of ``[N, M, K]``
tree storage at the game's current node, computes K PUCT scores in
registers, argmaxes, and writes the per-game outputs.

Why Triton (vs. ``torch.compile``):

- Inductor fuses what it can but still emits 3-5 distinct kernels per
  iteration; per-iteration launch overhead caps the speedup.
- Triton lets us pin the entire iteration body — gathers, PUCT compute,
  argmax, masked writes — into one kernel and amortize launch over all
  N games.
- Triton kernels also sidestep the in-place-mutation issues that defeat
  ``mode='reduce-overhead'`` cudagraphs.

Memory layout assumptions: tree tensors ``child_priors`` /
``child_visits`` / ``child_value_sum`` / ``child_rewards`` /
``child_actions`` / ``child_node_idx`` are contiguous ``[N, M, K]``;
``node_visits`` / ``path_node_idx`` / ``path_slot`` are contiguous
``[N, M]``; per-game state is contiguous ``[N]``. ``BLOCK_K`` must be a
power of 2 ≥ ``K`` (slots ≥ K are masked out as ``-inf`` in the score).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _puct_walk_kernel(
    # [N, M, K] tree storage (all element pointers).
    child_priors_ptr,        # float32
    child_visits_ptr,        # int32
    child_value_sum_ptr,     # float32
    child_rewards_ptr,       # float32
    child_actions_ptr,       # int32
    child_node_idx_ptr,      # int32
    # [N, M] node-level visits.
    node_visits_ptr,         # int32
    # [N, M] expected plies-to-end per node (MLH utility; read only when USE_ML).
    node_moves_left_ptr,     # float32
    # [N, M] path arrays — written at columns 0..MAX_DEPTH (root + each step).
    path_node_idx_ptr,       # int32
    path_slot_ptr,           # int32
    # [N] per-game state, write-only output.
    leaf_parent_node_ptr,    # int32 (= final cur)
    leaf_slot_ptr,           # int32
    leaf_action_ptr,         # int32
    depth_ptr,               # int32
    # [N] per-game scalars, read-only.
    mm_min_ptr,              # float32
    mm_max_ptr,              # float32
    # Constants (compile-time).
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    PB_C_BASE: tl.constexpr,
    PB_C_INIT: tl.constexpr,
    DISCOUNT: tl.constexpr,
    ML_SLOPE: tl.constexpr,
    ML_MAX_EFFECT: tl.constexpr,
    ML_THRESHOLD: tl.constexpr,
    USE_ML: tl.constexpr,
):
    """One full PUCT walk per game. ONE launch per simulation (vs. one per
    step), unrolling the MAX_DEPTH-step inner loop entirely on-device.

    Path arrays are write-only here (assumed pre-filled with -1; root
    written outside as ``path_node_idx[:, 0] = 0``); per-game state
    (current_node / still_walking / leaf_*) lives entirely in registers
    across the loop and is committed to memory only at the end.
    """
    g = tl.program_id(0)

    mm_min = tl.load(mm_min_ptr + g)
    mm_max = tl.load(mm_max_ptr + g)
    has_range = mm_max > mm_min
    span = tl.maximum(mm_max - mm_min, 1e-12)

    # Initial state — root, walking, no leaf info yet.
    cur = tl.zeros([], dtype=tl.int32)        # int32 scalar
    walking = tl.full([], 1, dtype=tl.int1)
    leaf_slot_v = tl.zeros([], dtype=tl.int32)
    leaf_action_v = tl.zeros([], dtype=tl.int32)
    depth_v = tl.zeros([], dtype=tl.int32)

    slot = tl.arange(0, BLOCK_K)
    valid_slot = slot < K

    for step in tl.static_range(MAX_DEPTH):
        cur64 = cur.to(tl.int64)
        nv_cur = tl.load(node_visits_ptr + g * M + cur64).to(tl.float32)
        base = g * (M * K) + cur64 * K

        priors     = tl.load(child_priors_ptr     + base + slot, mask=valid_slot, other=0.0)
        visits     = tl.load(child_visits_ptr     + base + slot, mask=valid_slot, other=0)
        value_sums = tl.load(child_value_sum_ptr  + base + slot, mask=valid_slot, other=0.0)
        rewards    = tl.load(child_rewards_ptr    + base + slot, mask=valid_slot, other=0.0)
        actions    = tl.load(child_actions_ptr    + base + slot, mask=valid_slot, other=-1)

        pb_c = tl.log((nv_cur + PB_C_BASE + 1.0) / PB_C_BASE) + PB_C_INIT
        sqrt_n = tl.sqrt(nv_cur)
        visits_f = visits.to(tl.float32)
        prior_score = pb_c * priors * sqrt_n / (1.0 + visits_f)

        child_value = tl.where(
            visits_f > 0, value_sums / tl.maximum(visits_f, 1.0), 0.0
        )
        raw_q = rewards - DISCOUNT * child_value
        q_norm = (raw_q - mm_min) / span
        normalized_q = tl.where(has_range, q_norm, raw_q)
        value_score = tl.where(visits_f > 0, normalized_q, 0.0)

        scores = prior_score + value_score

        # MLH utility — parity with the torch ``_select`` path. On EXPANDED
        # children (child_node_idx >= 0) with |Q| > threshold, nudge toward faster
        # wins / slower losses: scores += sign(-raw_q) * min(slope*child_m, max).
        # ``USE_ML`` is a constexpr → the whole block compiles out when disabled
        # (zero overhead on the non-MLH hot path).
        if USE_ML:
            child_idx_vec = tl.load(child_node_idx_ptr + base + slot, mask=valid_slot, other=-1)
            ml_valid = child_idx_vec >= 0
            safe_idx = tl.where(ml_valid, child_idx_vec, 0).to(tl.int64)
            child_m = tl.load(node_moves_left_ptr + g * M + safe_idx, mask=ml_valid, other=0.0)
            ml_gate = ml_valid & (tl.abs(raw_q) > ML_THRESHOLD)
            eff = tl.minimum(ML_SLOPE * child_m, ML_MAX_EFFECT)
            sgn = tl.where(raw_q > 0, -1.0, tl.where(raw_q < 0, 1.0, 0.0))  # = sign(-raw_q)
            ml_term = tl.where(ml_gate, sgn * eff, 0.0)
            scores = scores + ml_term

        invalid = (actions == -1) | (~valid_slot)
        scores = tl.where(invalid, float("-inf"), scores)

        next_slot = tl.argmax(scores, axis=0)
        next_slot_i32 = next_slot.to(tl.int32)
        next_slot_i64 = next_slot.to(tl.int64)

        chosen_action = tl.load(child_actions_ptr   + base + next_slot_i64)
        chosen_child  = tl.load(child_node_idx_ptr  + base + next_slot_i64)

        unexpanded = chosen_child == -1
        stops_now = walking & unexpanded
        advance = walking & (~unexpanded)

        # Capture leaf info (only the first stop wins because stops_now is
        # True only on the iteration that first sees an unexpanded slot).
        leaf_slot_v = tl.where(stops_now, next_slot_i32, leaf_slot_v)
        leaf_action_v = tl.where(stops_now, chosen_action, leaf_action_v)
        depth_v = tl.where(stops_now, step, depth_v)

        # Path writes at column step+1.
        path_off = g * M + (step + 1)
        tl.store(
            path_node_idx_ptr + path_off,
            tl.where(advance, chosen_child, -1).to(tl.int32),
        )
        tl.store(
            path_slot_ptr + path_off,
            tl.where(advance, next_slot_i32, -1),
        )

        cur = tl.where(advance, chosen_child, cur)
        walking = advance

    # If a game is still walking past MAX_DEPTH (rare), force-stop at the
    # last visited node. The leaf slot/action are taken from the most
    # recent step's chosen values via leaf_slot_v / leaf_action_v which
    # were never updated (still default 0). Treat depth as MAX_DEPTH-1.
    leaf_slot_v = tl.where(walking, next_slot_i32, leaf_slot_v)
    leaf_action_v = tl.where(walking, chosen_action, leaf_action_v)
    depth_v = tl.where(walking, MAX_DEPTH - 1, depth_v)

    tl.store(leaf_parent_node_ptr + g, cur)
    tl.store(leaf_slot_ptr + g, leaf_slot_v)
    tl.store(leaf_action_ptr + g, leaf_action_v)
    tl.store(depth_ptr + g, depth_v)


def puct_walk(
    max_depth: int,
    *,
    child_priors: torch.Tensor,
    child_visits: torch.Tensor,
    child_value_sum: torch.Tensor,
    child_rewards: torch.Tensor,
    child_actions: torch.Tensor,
    child_node_idx: torch.Tensor,
    node_visits: torch.Tensor,
    node_moves_left: torch.Tensor,
    path_node_idx: torch.Tensor,
    path_slot: torch.Tensor,
    leaf_parent_node: torch.Tensor,
    leaf_slot: torch.Tensor,
    leaf_action: torch.Tensor,
    depth: torch.Tensor,
    mm_min: torch.Tensor,
    mm_max: torch.Tensor,
    pb_c_base: int,
    pb_c_init: float,
    discount: float,
    ml_slope: float = 0.0,
    ml_max_effect: float = 0.0,
    ml_threshold: float = 0.0,
    use_ml: bool = False,
) -> None:
    """Run a full PUCT walk (up to max_depth) for all N games via Triton.

    ONE kernel launch handles all max_depth steps for all games — the
    inner depth loop is unrolled inside the kernel. Caller must pre-fill
    ``path_node_idx`` and ``path_slot`` with -1 sentinels (and root at
    column 0) before this call.

    Outputs (written): ``leaf_parent_node``, ``leaf_slot``, ``leaf_action``,
    ``depth``, plus ``path_node_idx[:, 1..max_depth]`` and
    ``path_slot[:, 1..max_depth]``.
    """
    n, m, k = child_priors.shape
    assert node_moves_left.shape == (n, m)
    assert path_node_idx.shape == (n, m)
    assert path_slot.shape == (n, m)
    assert leaf_parent_node.shape == (n,) and leaf_parent_node.dtype == torch.int32
    assert leaf_slot.shape == (n,) and leaf_slot.dtype == torch.int32
    assert leaf_action.shape == (n,) and leaf_action.dtype == torch.int32
    assert depth.shape == (n,) and depth.dtype == torch.int32
    assert max_depth + 1 <= m, "max_depth + 1 must fit in path arrays"

    block_k = next_power_of_2(k)
    grid = (n,)
    _puct_walk_kernel[grid](
        child_priors,
        child_visits,
        child_value_sum,
        child_rewards,
        child_actions,
        child_node_idx,
        node_visits,
        node_moves_left,
        path_node_idx,
        path_slot,
        leaf_parent_node,
        leaf_slot,
        leaf_action,
        depth,
        mm_min,
        mm_max,
        M=m,
        K=k,
        BLOCK_K=block_k,
        MAX_DEPTH=max_depth,
        PB_C_BASE=pb_c_base,
        PB_C_INIT=pb_c_init,
        DISCOUNT=discount,
        ML_SLOPE=ml_slope,
        ML_MAX_EFFECT=ml_max_effect,
        ML_THRESHOLD=ml_threshold,
        USE_ML=use_ml,
    )
