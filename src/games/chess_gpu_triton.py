"""Triton kernels for ``chess_gpu.py`` hot paths.

Currently provides a fused 8-direction slider-attacks kernel. The PyTorch
implementation calls ``_slider_attacks`` once per direction; in
``_compute_pin_filter`` (16 calls), ``_compute_check_resolve`` (8 calls),
and ``attacks_by_color`` (4 calls) the per-call launch overhead dominates
the actual Kogge-Stone fill compute. This kernel computes all 8 directions
in ONE launch, with all per-direction state held in registers.

Triton has no native logical right shift on int64 (``>>`` sign-extends).
We emulate via mask: ``LSR(x, n) = (x >> n) & ((1 << (64-n)) - 1)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Direction layout MUST match chess_gpu.py:
#   DIR_N=0, DIR_NE=1, DIR_E=2, DIR_SE=3, DIR_S=4, DIR_SW=5, DIR_W=6, DIR_NW=7
# The output [N, 8] uses this column ordering so callers indexing by
# ``out[:, DIR_X]`` get the correct attack bitboard.

# File-wrap masks (no-cross-boundary).
_NOT_FILE_A = 0xFEFEFEFEFEFEFEFE - (1 << 64)  # signed int64 representation
_NOT_FILE_H = 0x7F7F7F7F7F7F7F7F


@triton.jit
def _slider_8way_kernel(
    seed_ptr,           # int64 [N]
    empty_ptr,          # int64 [N]
    out_ptr,            # int64 [N, 8] (row-major)
    n_games,            # int
    NOT_FILE_A: tl.constexpr,
    NOT_FILE_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute Kogge-Stone slider attacks for all 8 directions per game.

    All directions share the same ``empty`` mask. Each program handles
    BLOCK_N consecutive games. Output layout: out[g, dir] for dir in
    DIR_N..DIR_NW (matching chess_gpu.py's enum).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_games

    seed = tl.load(seed_ptr + offs, mask=mask, other=0)
    empty = tl.load(empty_ptr + offs, mask=mask, other=0)

    # Logical-right-shift mask constants (int64 representation of the
    # post-shift mask `(1 << (64 - n)) - 1`).
    M_56 = (1 << 56) - 1
    M_48 = (1 << 48) - 1
    M_32 = (1 << 32) - 1
    M_57 = (1 << 57) - 1
    M_50 = (1 << 50) - 1
    M_36 = (1 << 36) - 1
    M_55 = (1 << 55) - 1
    M_46 = (1 << 46) - 1
    M_28 = (1 << 28) - 1
    M_63 = (1 << 63) - 1
    M_62 = (1 << 62) - 1
    M_60 = (1 << 60) - 1

    # ----------- DIR_N (<< 8, no file mask) -----------
    g = seed; e = empty
    g = g | (e & (g << 8));  e = e & (e << 8)
    g = g | (e & (g << 16)); e = e & (e << 16)
    g = g | (e & (g << 32))
    out_n = g << 8

    # ----------- DIR_NE (<< 9, NOT_FILE_A) -----------
    g = seed; e = empty & NOT_FILE_A
    g = g | (e & (g << 9));  e = e & (e << 9)
    g = g | (e & (g << 18)); e = e & (e << 18)
    g = g | (e & (g << 36))
    out_ne = (g << 9) & NOT_FILE_A

    # ----------- DIR_E (<< 1, NOT_FILE_A) -----------
    g = seed; e = empty & NOT_FILE_A
    g = g | (e & (g << 1)); e = e & (e << 1)
    g = g | (e & (g << 2)); e = e & (e << 2)
    g = g | (e & (g << 4))
    out_e = (g << 1) & NOT_FILE_A

    # ----------- DIR_SE (>> 7 logical, NOT_FILE_A) -----------
    g = seed; e = empty & NOT_FILE_A
    g = g | (e & ((g >> 7) & M_57));  e = e & ((e >> 7) & M_57)
    g = g | (e & ((g >> 14) & M_50)); e = e & ((e >> 14) & M_50)
    g = g | (e & ((g >> 28) & M_36))
    out_se = ((g >> 7) & M_57) & NOT_FILE_A

    # ----------- DIR_S (>> 8 logical, no file mask) -----------
    g = seed; e = empty
    g = g | (e & ((g >> 8) & M_56));  e = e & ((e >> 8) & M_56)
    g = g | (e & ((g >> 16) & M_48)); e = e & ((e >> 16) & M_48)
    g = g | (e & ((g >> 32) & M_32))
    out_s = (g >> 8) & M_56

    # ----------- DIR_SW (>> 9 logical, NOT_FILE_H) -----------
    g = seed; e = empty & NOT_FILE_H
    g = g | (e & ((g >> 9) & M_55));  e = e & ((e >> 9) & M_55)
    g = g | (e & ((g >> 18) & M_46)); e = e & ((e >> 18) & M_46)
    g = g | (e & ((g >> 36) & M_28))
    out_sw = ((g >> 9) & M_55) & NOT_FILE_H

    # ----------- DIR_W (>> 1 logical, NOT_FILE_H) -----------
    g = seed; e = empty & NOT_FILE_H
    g = g | (e & ((g >> 1) & M_63)); e = e & ((e >> 1) & M_63)
    g = g | (e & ((g >> 2) & M_62)); e = e & ((e >> 2) & M_62)
    g = g | (e & ((g >> 4) & M_60))
    out_w = ((g >> 1) & M_63) & NOT_FILE_H

    # ----------- DIR_NW (<< 7, NOT_FILE_H) -----------
    g = seed; e = empty & NOT_FILE_H
    g = g | (e & (g << 7));  e = e & (e << 7)
    g = g | (e & (g << 14)); e = e & (e << 14)
    g = g | (e & (g << 28))
    out_nw = (g << 7) & NOT_FILE_H

    # Store all 8 directions in declared enum order.
    base = offs * 8
    tl.store(out_ptr + base + 0, out_n,  mask=mask)
    tl.store(out_ptr + base + 1, out_ne, mask=mask)
    tl.store(out_ptr + base + 2, out_e,  mask=mask)
    tl.store(out_ptr + base + 3, out_se, mask=mask)
    tl.store(out_ptr + base + 4, out_s,  mask=mask)
    tl.store(out_ptr + base + 5, out_sw, mask=mask)
    tl.store(out_ptr + base + 6, out_w,  mask=mask)
    tl.store(out_ptr + base + 7, out_nw, mask=mask)


def slider_attacks_8way(seed: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    """Return [N, 8] int64 — slider attacks per direction (DIR_N..DIR_NW).

    All 8 directions share ``empty[N]``. Replaces 8 separate calls to the
    PyTorch ``_slider_attacks`` (one per direction) with one Triton launch.

    Inputs are forced contiguous: the kernel uses ``tl.load(ptr + offs)``
    which assumes element-stride-1 layout. Callers that pass column slices
    of wider tensors (e.g. ``state.pieces[:, P_KING]`` has stride 6) would
    otherwise read interleaved memory and silently produce wrong attack
    bitboards for batch indices > 0 (idx 0 reads correctly because offset 0
    is stride-invariant; idx 1+ get garbage). Bug found 2026-05-08;
    materialized as ~50% of self-play games training on chess-illegal
    positions during the 2026_05_07 runs.
    """
    assert seed.dtype == torch.int64 and empty.dtype == torch.int64
    assert seed.is_cuda and empty.is_cuda
    seed = seed.contiguous()
    empty = empty.contiguous()
    n = seed.shape[0]
    out = torch.empty((n, 8), dtype=torch.int64, device=seed.device)
    BLOCK_N = 64
    grid = (triton.cdiv(n, BLOCK_N),)
    _slider_8way_kernel[grid](
        seed,
        empty,
        out,
        n,
        NOT_FILE_A=_NOT_FILE_A,
        NOT_FILE_H=_NOT_FILE_H,
        BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit
def _slider_8way_per_dir_empty_kernel(
    seed_ptr,           # int64 [N]
    empties_ptr,        # int64 [N, 8] — one empty mask per direction
    out_ptr,            # int64 [N, 8]
    n_games,
    NOT_FILE_A: tl.constexpr,
    NOT_FILE_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Like _slider_8way_kernel but each direction has its own ``empty[N, d]``.

    Used by ``_compute_pin_filter`` second pass, where each direction's
    empty mask differs (it's the original empty XOR the first-blocker bit
    for that direction).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_games

    seed = tl.load(seed_ptr + offs, mask=mask, other=0)

    M_56 = (1 << 56) - 1
    M_48 = (1 << 48) - 1
    M_32 = (1 << 32) - 1
    M_57 = (1 << 57) - 1
    M_50 = (1 << 50) - 1
    M_36 = (1 << 36) - 1
    M_55 = (1 << 55) - 1
    M_46 = (1 << 46) - 1
    M_28 = (1 << 28) - 1
    M_63 = (1 << 63) - 1
    M_62 = (1 << 62) - 1
    M_60 = (1 << 60) - 1

    base_e = offs * 8
    base_o = offs * 8

    # DIR_N (col 0).
    e_n = tl.load(empties_ptr + base_e + 0, mask=mask, other=0)
    g = seed; e = e_n
    g = g | (e & (g << 8));  e = e & (e << 8)
    g = g | (e & (g << 16)); e = e & (e << 16)
    g = g | (e & (g << 32))
    tl.store(out_ptr + base_o + 0, g << 8, mask=mask)

    # DIR_NE (col 1).
    e_ne = tl.load(empties_ptr + base_e + 1, mask=mask, other=0) & NOT_FILE_A
    g = seed; e = e_ne
    g = g | (e & (g << 9));  e = e & (e << 9)
    g = g | (e & (g << 18)); e = e & (e << 18)
    g = g | (e & (g << 36))
    tl.store(out_ptr + base_o + 1, (g << 9) & NOT_FILE_A, mask=mask)

    # DIR_E (col 2).
    e_e = tl.load(empties_ptr + base_e + 2, mask=mask, other=0) & NOT_FILE_A
    g = seed; e = e_e
    g = g | (e & (g << 1)); e = e & (e << 1)
    g = g | (e & (g << 2)); e = e & (e << 2)
    g = g | (e & (g << 4))
    tl.store(out_ptr + base_o + 2, (g << 1) & NOT_FILE_A, mask=mask)

    # DIR_SE (col 3).
    e_se = tl.load(empties_ptr + base_e + 3, mask=mask, other=0) & NOT_FILE_A
    g = seed; e = e_se
    g = g | (e & ((g >> 7) & M_57));  e = e & ((e >> 7) & M_57)
    g = g | (e & ((g >> 14) & M_50)); e = e & ((e >> 14) & M_50)
    g = g | (e & ((g >> 28) & M_36))
    tl.store(out_ptr + base_o + 3, ((g >> 7) & M_57) & NOT_FILE_A, mask=mask)

    # DIR_S (col 4).
    e_s = tl.load(empties_ptr + base_e + 4, mask=mask, other=0)
    g = seed; e = e_s
    g = g | (e & ((g >> 8) & M_56));  e = e & ((e >> 8) & M_56)
    g = g | (e & ((g >> 16) & M_48)); e = e & ((e >> 16) & M_48)
    g = g | (e & ((g >> 32) & M_32))
    tl.store(out_ptr + base_o + 4, (g >> 8) & M_56, mask=mask)

    # DIR_SW (col 5).
    e_sw = tl.load(empties_ptr + base_e + 5, mask=mask, other=0) & NOT_FILE_H
    g = seed; e = e_sw
    g = g | (e & ((g >> 9) & M_55));  e = e & ((e >> 9) & M_55)
    g = g | (e & ((g >> 18) & M_46)); e = e & ((e >> 18) & M_46)
    g = g | (e & ((g >> 36) & M_28))
    tl.store(out_ptr + base_o + 5, ((g >> 9) & M_55) & NOT_FILE_H, mask=mask)

    # DIR_W (col 6).
    e_w = tl.load(empties_ptr + base_e + 6, mask=mask, other=0) & NOT_FILE_H
    g = seed; e = e_w
    g = g | (e & ((g >> 1) & M_63)); e = e & ((e >> 1) & M_63)
    g = g | (e & ((g >> 2) & M_62)); e = e & ((e >> 2) & M_62)
    g = g | (e & ((g >> 4) & M_60))
    tl.store(out_ptr + base_o + 6, ((g >> 1) & M_63) & NOT_FILE_H, mask=mask)

    # DIR_NW (col 7).
    e_nw = tl.load(empties_ptr + base_e + 7, mask=mask, other=0) & NOT_FILE_H
    g = seed; e = e_nw
    g = g | (e & (g << 7));  e = e & (e << 7)
    g = g | (e & (g << 14)); e = e & (e << 14)
    g = g | (e & (g << 28))
    tl.store(out_ptr + base_o + 7, (g << 7) & NOT_FILE_H, mask=mask)


def slider_attacks_8way_per_dir_empty(
    seed: torch.Tensor, empties: torch.Tensor
) -> torch.Tensor:
    """Return [N, 8] int64 — slider attacks per direction with per-direction
    ``empties[N, 8]``. Used by pin-filter x-ray pass.

    Both inputs are forced contiguous — see ``slider_attacks_8way`` docstring
    for the rationale. The previous version called ``.contiguous()`` on
    ``empties`` only; ``seed`` was missing the same guard, so column-sliced
    seeds (the king bitboard is sliced from the (N, 12) pieces tensor with
    stride 12) produced wrong output for batch indices > 0.
    """
    assert seed.dtype == torch.int64 and empties.dtype == torch.int64
    assert seed.is_cuda and empties.is_cuda
    assert empties.shape[1] == 8
    seed = seed.contiguous()
    empties = empties.contiguous()
    n = seed.shape[0]
    out = torch.empty((n, 8), dtype=torch.int64, device=seed.device)
    BLOCK_N = 64
    grid = (triton.cdiv(n, BLOCK_N),)
    _slider_8way_per_dir_empty_kernel[grid](
        seed,
        empties,
        out,
        n,
        NOT_FILE_A=_NOT_FILE_A,
        NOT_FILE_H=_NOT_FILE_H,
        BLOCK_N=BLOCK_N,
    )
    return out
