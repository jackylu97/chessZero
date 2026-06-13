"""GPU-friendly batched chess implementation (bitboard backend).

Phase 1: state representation, reset_batch, from_python_chess, to_tensor_batch.
Phase 2: legal_mask.
Phase 3: step_batch (incl. terminals + repetition detection).
Phase 4: torch.compile + CUDA residency.

Action-index space is identical to ChessGame's (`64 * 73 = 4672`); see
`chess.py:_move_to_action` for the encoding.

Bitboard convention: bit `i` ↔ square `i` ↔ (rank `i//8`, file `i%8`).
This matches python-chess and the row/col layout of `ChessGame.to_tensor`.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import torch

from .batched_base import BatchedGame, BatchedGameState
from .chess import ACTION_SPACE


# -- piece / plane indexing --------------------------------------------------
# Plane-layout constants must match `ChessGame.to_tensor`. See chess.py:163-197.
NUM_PLANES = 19

# Castling-rights bit positions in the (N, 4) bool tensor.
CR_WK, CR_WQ, CR_BK, CR_BQ = 0, 1, 2, 3

# python-chess piece-type → 0..5 plane offset (per color). White uses base 0,
# black uses base 6 in `pieces`.
_PIECE_TYPE_TO_IDX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Plane indices into ChessBatchedState.pieces. White at base 0, black at base 6.
P_PAWN, P_KNIGHT, P_BISHOP, P_ROOK, P_QUEEN, P_KING = 0, 1, 2, 3, 4, 5


# -- bitboard helpers --------------------------------------------------------
def _u64_to_int64_tensor(arr: np.ndarray) -> torch.Tensor:
    """Reinterpret a uint64 numpy array as int64 with the same bit pattern.

    Used everywhere bitboards cross from numpy/python ints (which are arbitrary
    precision) into torch (which only has signed int64). Bit-level ops on the
    result are bit-identical to the unsigned interpretation; sign-extension on
    arithmetic right-shift is harmless because we always strip the high bits
    with `& 1` (or via shift-down-then-AND-mask).
    """
    return torch.from_numpy(arr.view(np.int64).copy())


def _bit(sq: int) -> np.uint64:
    return np.uint64(1) << np.uint64(sq)


# File / rank constants for shift wrap-masking. Stored as int64 with the
# uint64 bit pattern (see _u64_to_int64_tensor).
def _build_file_mask(file: int) -> int:
    bb = np.uint64(0)
    for r in range(8):
        bb |= _bit(r * 8 + file)
    return int(bb.view(np.int64))


FILE_A = _build_file_mask(0)
FILE_H = _build_file_mask(7)
NOT_FILE_A = ~FILE_A  # for east shifts (<< 1, << 9, >> 7)
NOT_FILE_H = ~FILE_H  # for west shifts (>> 1, >> 9, << 7)

# Square-color masks (bit patterns match python-chess BB_DARK/LIGHT_SQUARES),
# stored as int64. Used by _insufficient_material (bug_hunt_2026_06_13.md §C).
BB_DARK_SQUARES = int(np.uint64(0xaa55aa55aa55aa55).view(np.int64))
BB_LIGHT_SQUARES = int(np.uint64(0x55aa55aa55aa55aa).view(np.int64))


def _lsr(g: torch.Tensor, n: int) -> torch.Tensor:
    """Logical (unsigned) right shift on int64 tensor.

    PyTorch's ``>>`` on int64 is arithmetic — it sign-extends from bit 63. For
    bitboards with bit 63 set (h8 occupied) this leaks phantom high bits and
    corrupts slider attack generation. Mask off the sign-extended bits to get
    the unsigned-equivalent shift.
    """
    if n <= 0:
        return g
    mask = (1 << (64 - n)) - 1
    return (g >> n) & mask


# -- lookup tables (module-level, computed once at import) -------------------
def _build_knight_attacks() -> torch.Tensor:
    out = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, c = divmod(sq, 8)
        for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                out[sq] |= _bit(nr * 8 + nc)
    return _u64_to_int64_tensor(out)


def _build_king_attacks() -> torch.Tensor:
    out = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, c = divmod(sq, 8)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    out[sq] |= _bit(nr * 8 + nc)
    return _u64_to_int64_tensor(out)


def _build_pawn_attacks(color: int) -> torch.Tensor:
    """Diagonal forward attacks for a pawn of `color` (0=white, 1=black)."""
    out = np.zeros(64, dtype=np.uint64)
    dr = 1 if color == 0 else -1
    for sq in range(64):
        r, c = divmod(sq, 8)
        for dc in (-1, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                out[sq] |= _bit(nr * 8 + nc)
    return _u64_to_int64_tensor(out)


# Direction order: N, NE, E, SE, S, SW, W, NW. Used for ray indexing.
DIR_N, DIR_NE, DIR_E, DIR_SE, DIR_S, DIR_SW, DIR_W, DIR_NW = range(8)
_DIR_DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def _build_rays() -> torch.Tensor:
    """RAYS[d, sq] = bitboard of squares the ray from `sq` covers in direction `d`,
    NOT including `sq` itself.
    """
    out = np.zeros((8, 64), dtype=np.uint64)
    for d, (dr, dc) in enumerate(_DIR_DELTAS):
        for sq in range(64):
            r, c = divmod(sq, 8)
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                out[d, sq] |= _bit(nr * 8 + nc)
                nr += dr
                nc += dc
    return _u64_to_int64_tensor(out)


KNIGHT_ATTACKS = _build_knight_attacks()
KING_ATTACKS = _build_king_attacks()
PAWN_ATTACKS_W = _build_pawn_attacks(0)
PAWN_ATTACKS_B = _build_pawn_attacks(1)
RAYS = _build_rays()


# -- vectorized attack-set helpers -------------------------------------------
def _from_sq_attacks(piece_bb: torch.Tensor, attack_table: torch.Tensor) -> torch.Tensor:
    """Union of attack bitboards from every set square of `piece_bb`.

    Args:
        piece_bb: (N,) int64 — bitboard of piece locations per game.
        attack_table: (64,) int64 — per-square attack bitboard.

    Returns:
        (N,) int64 — OR of `attack_table[sq]` over all `sq` set in `piece_bb`.

    Vectorized: build the (N, 64) per-square contribution tensor in one
    pass, then tree-OR-reduce along the 64-axis in O(log 64) = 6 passes.
    Replaces the previous ``for sq in range(64)`` which fired 64 small
    GPU launches per call — and this function gets called 3× per
    ``attacks_by_color`` (pawn / knight / king) × multiple times per
    legal_mask.
    """
    N = piece_bb.shape[0]
    dev = piece_bb.device
    sq_idx = torch.arange(64, device=dev, dtype=torch.int64)

    # on_sq[i, s] = piece sits at square s for game i.
    on_sq = ((piece_bb.unsqueeze(1) >> sq_idx) & 1).bool()    # (N, 64)
    # contributions[i, s] = attack_table[s] if on_sq else 0.
    contrib = torch.where(
        on_sq,
        attack_table.unsqueeze(0),
        torch.zeros((), dtype=torch.int64, device=dev),
    )                                                          # (N, 64)
    # Tree reduce: 64 → 32 → 16 → 8 → 4 → 2 → 1 via half-and-OR.
    while contrib.shape[1] > 1:
        size = contrib.shape[1]
        if size % 2 == 1:
            pad = torch.zeros(N, 1, dtype=torch.int64, device=dev)
            contrib = torch.cat([contrib, pad], dim=1)
            size += 1
        half = size // 2
        contrib = contrib[:, :half] | contrib[:, half:]
    return contrib.squeeze(1)


def _slider_attacks(sliders: torch.Tensor, empty: torch.Tensor,
                    direction: int) -> torch.Tensor:
    """Kogge-Stone fill: union of slider attack bitboards in one direction.

    Args:
        sliders: (N,) int64 — bitboard of slider piece locations.
        empty:   (N,) int64 — bitboard of empty squares.
        direction: one of DIR_N..DIR_NW.

    Returns: (N,) int64 — squares attacked along `direction`. The first blocker
    (own or enemy) is included in the attack set; squares past it are not.
    The slider's own square is NOT included.
    """
    # Per-direction (shift, file_wrap_mask) pairs. Wrap masks ensure shifts
    # don't bleed across the a/h-file boundary.
    # Use `g = g | ...` (binding-rebind) instead of `g |= ...` (in-place):
    # `g = sliders` aliases the input; an in-place OR would mutate the caller's
    # tensor, corrupting subsequent direction calls in `_rook_attacks` /
    # `_bishop_attacks` that share the same sliders argument.
    g = sliders
    e = empty
    if direction == DIR_N:        # shift << 8 (no file wrap)
        g = g | (e & (g << 8));  e = e & (e << 8)
        g = g | (e & (g << 16)); e = e & (e << 16)
        g = g | (e & (g << 32))
        return (g << 8)
    if direction == DIR_S:        # shift >> 8 (logical)
        g = g | (e & _lsr(g, 8));  e = e & _lsr(e, 8)
        g = g | (e & _lsr(g, 16)); e = e & _lsr(e, 16)
        g = g | (e & _lsr(g, 32))
        return _lsr(g, 8)
    if direction == DIR_E:        # shift << 1, mask not_a_file
        m = NOT_FILE_A
        e = e & m
        g = g | (e & (g << 1)); e = e & (e << 1)
        g = g | (e & (g << 2)); e = e & (e << 2)
        g = g | (e & (g << 4))
        return ((g << 1) & m)
    if direction == DIR_W:        # shift >> 1 (logical), mask not_h_file
        m = NOT_FILE_H
        e = e & m
        g = g | (e & _lsr(g, 1)); e = e & _lsr(e, 1)
        g = g | (e & _lsr(g, 2)); e = e & _lsr(e, 2)
        g = g | (e & _lsr(g, 4))
        return _lsr(g, 1) & m
    if direction == DIR_NE:       # shift << 9, mask not_a_file
        m = NOT_FILE_A
        e = e & m
        g = g | (e & (g << 9));  e = e & (e << 9)
        g = g | (e & (g << 18)); e = e & (e << 18)
        g = g | (e & (g << 36))
        return ((g << 9) & m)
    if direction == DIR_NW:       # shift << 7, mask not_h_file
        m = NOT_FILE_H
        e = e & m
        g = g | (e & (g << 7));  e = e & (e << 7)
        g = g | (e & (g << 14)); e = e & (e << 14)
        g = g | (e & (g << 28))
        return ((g << 7) & m)
    if direction == DIR_SE:       # shift >> 7 (logical), mask not_a_file
        m = NOT_FILE_A
        e = e & m
        g = g | (e & _lsr(g, 7));  e = e & _lsr(e, 7)
        g = g | (e & _lsr(g, 14)); e = e & _lsr(e, 14)
        g = g | (e & _lsr(g, 28))
        return _lsr(g, 7) & m
    if direction == DIR_SW:       # shift >> 9 (logical), mask not_h_file
        m = NOT_FILE_H
        e = e & m
        g = g | (e & _lsr(g, 9));  e = e & _lsr(e, 9)
        g = g | (e & _lsr(g, 18)); e = e & _lsr(e, 18)
        g = g | (e & _lsr(g, 36))
        return _lsr(g, 9) & m
    raise ValueError(f"unknown direction {direction}")


_ROOK_DIRS = (DIR_N, DIR_E, DIR_S, DIR_W)
_BISHOP_DIRS = (DIR_NE, DIR_SE, DIR_SW, DIR_NW)


def _rook_attacks(sliders: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(sliders)
    for d in _ROOK_DIRS:
        out = out | _slider_attacks(sliders, empty, d)
    return out


def _bishop_attacks(sliders: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(sliders)
    for d in _BISHOP_DIRS:
        out = out | _slider_attacks(sliders, empty, d)
    return out


def attacks_by_color(state: "ChessBatchedState", color: int,
                     occupancy: torch.Tensor | None = None) -> torch.Tensor:
    """Union bitboard of all squares attacked by `color` (0=white, 1=black).

    Pawns count diagonal attacks only (not pushes — pushes are not attacks).
    Sliders pass through `occupancy` blockers; the first blocker square is
    included in the attack set.

    `occupancy` defaults to `our + opp`. Pass a custom value to compute
    attacks "through" specific pieces — e.g. king-x-ray for legality
    (compute opp attacks with our king masked out so its presence doesn't
    block opponent slider rays through it).
    """
    base = 0 if color == 0 else 6
    pawns = state.pieces[:, base + P_PAWN]
    knights = state.pieces[:, base + P_KNIGHT]
    bishops = state.pieces[:, base + P_BISHOP]
    rooks = state.pieces[:, base + P_ROOK]
    queens = state.pieces[:, base + P_QUEEN]
    king = state.pieces[:, base + P_KING]

    if occupancy is None:
        all_pieces = state.pieces.sum(dim=1) * 0  # placeholder (unused)
        # Build occupancy from all 12 piece bitboards via OR-reduce.
        occupancy = torch.zeros_like(pawns)
        for p in range(12):
            occupancy = occupancy | state.pieces[:, p]
    empty = ~occupancy

    pawn_table = PAWN_ATTACKS_W if color == 0 else PAWN_ATTACKS_B
    pawn_table = pawn_table.to(state.device)
    knight_table = KNIGHT_ATTACKS.to(state.device)
    king_table = KING_ATTACKS.to(state.device)

    pawn_a = _from_sq_attacks(pawns, pawn_table)
    knight_a = _from_sq_attacks(knights, knight_table)
    king_a = _from_sq_attacks(king, king_table)
    # Slider attacks via fused 8-direction Triton kernel: one launch covers
    # all 8 ray directions for both bishop+queen (diagonal subset) and
    # rook+queen (orthogonal subset). Replaces 8 separate _slider_attacks
    # calls with one — see chess_gpu_triton.py.
    if state.device.type == "cuda":
        from src.games.chess_gpu_triton import slider_attacks_8way
        bq_attacks = slider_attacks_8way((bishops | queens), empty)   # (N, 8)
        rq_attacks = slider_attacks_8way((rooks | queens), empty)     # (N, 8)
        # Bishops use diagonals: NE, SE, SW, NW (= cols 1, 3, 5, 7).
        # Rooks use orthogonals: N, E, S, W (= cols 0, 2, 4, 6).
        bishop_a = bq_attacks[:, 1] | bq_attacks[:, 3] | bq_attacks[:, 5] | bq_attacks[:, 7]
        rook_a = rq_attacks[:, 0] | rq_attacks[:, 2] | rq_attacks[:, 4] | rq_attacks[:, 6]
    else:
        bishop_a = _bishop_attacks(bishops | queens, empty)
        rook_a = _rook_attacks(rooks | queens, empty)
    return pawn_a | knight_a | king_a | bishop_a | rook_a


# -- action-index encoding tables --------------------------------------------
# Action layout (matches chess.py:_move_to_action exactly):
#   move_type ∈ [0, 56): queen-type (8 dirs × 7 distances).
#   move_type ∈ [56, 64): knight-type (8 knight offsets).
#   move_type ∈ [64, 73): underpromotion (3 promo pieces × 3 directions).
# Action index = from_sq * 73 + move_type.

# Queen-type direction → (dr, dc). Order matches chess.py's encoder.
_ACTION_QUEEN_DIRS = (
    (0, 1),    # 0: E
    (1, 1),    # 1: NE
    (1, 0),    # 2: N
    (1, -1),   # 3: NW
    (0, -1),   # 4: W
    (-1, -1),  # 5: SW
    (-1, 0),   # 6: S
    (-1, 1),   # 7: SE
)

# Knight offsets in chess.py order (m=56..63).
_ACTION_KNIGHT_DELTAS = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)

# Underpromotion (m=64..72): 3 promo pieces × 3 captures.
# direction: 0=straight (dc=0), 1=right (dc=+1), 2=left (dc=-1) per chess.py.
# promo: 0=ROOK, 1=BISHOP, 2=KNIGHT.
_UNDERPROMO_DC = (0, 1, -1)   # repeated 3 times across promo pieces


def _build_action_tables() -> dict:
    """Precompute per-action (from_sq, to_sq, kind, promo) tables.

    Two `target_*` arrays are built, one per side, because underpromos use
    `dr = +1` (white) or `dr = -1` (black). Queen-type and knight-type are
    color-independent so the two arrays match for those entries.

    Returns a dict of int64 arrays of length 4672:
        target_w / target_b: to_sq, or -1 if off-board.
        is_queen_type / is_knight_type / is_underpromo: bool flags.
        promo_piece: 0=R, 1=B, 2=N for underpromos, else -1.
        from_sq: idx // 73 (precomputed for convenience).
    """
    target_w = np.full(4672, -1, dtype=np.int64)
    target_b = np.full(4672, -1, dtype=np.int64)
    is_queen = np.zeros(4672, dtype=bool)
    is_knight = np.zeros(4672, dtype=bool)
    is_underpromo = np.zeros(4672, dtype=bool)
    promo_piece = np.full(4672, -1, dtype=np.int64)
    from_sq_arr = np.zeros(4672, dtype=np.int64)

    for from_sq in range(64):
        r, f = divmod(from_sq, 8)
        for mt in range(73):
            idx = from_sq * 73 + mt
            from_sq_arr[idx] = from_sq

            if mt < 56:
                # Queen-type.
                d = mt // 7
                dist = mt % 7 + 1
                dr, dc = _ACTION_QUEEN_DIRS[d]
                nr, nc = r + dr * dist, f + dc * dist
                if 0 <= nr < 8 and 0 <= nc < 8:
                    target_w[idx] = nr * 8 + nc
                    target_b[idx] = nr * 8 + nc
                is_queen[idx] = True
            elif mt < 64:
                # Knight-type.
                dr, dc = _ACTION_KNIGHT_DELTAS[mt - 56]
                nr, nc = r + dr, f + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    target_w[idx] = nr * 8 + nc
                    target_b[idx] = nr * 8 + nc
                is_knight[idx] = True
            else:
                # Underpromotion.
                ui = mt - 64
                dc = _UNDERPROMO_DC[ui % 3]
                # White: dr=+1; Black: dr=-1.
                for dr, target_arr in ((1, target_w), (-1, target_b)):
                    nr = r + dr
                    nc = f + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        target_arr[idx] = nr * 8 + nc
                is_underpromo[idx] = True
                promo_piece[idx] = ui // 3  # 0=R, 1=B, 2=N

    return {
        "target_w": torch.from_numpy(target_w),
        "target_b": torch.from_numpy(target_b),
        "is_queen_type": torch.from_numpy(is_queen),
        "is_knight_type": torch.from_numpy(is_knight),
        "is_underpromo": torch.from_numpy(is_underpromo),
        "promo_piece": torch.from_numpy(promo_piece),
        "from_sq": torch.from_numpy(from_sq_arr),
    }


_ACTION_TABLES = _build_action_tables()
ACTION_TARGET_W = _ACTION_TABLES["target_w"]               # (4672,) int64
ACTION_TARGET_B = _ACTION_TABLES["target_b"]               # (4672,) int64
ACTION_IS_UNDERPROMO = _ACTION_TABLES["is_underpromo"]     # (4672,) bool
ACTION_FROM_SQ = _ACTION_TABLES["from_sq"]                 # (4672,) int64


# -- legal_mask --------------------------------------------------------------
def _or_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Reduce along `dim` via bitwise OR. Torch lacks `bitwise_or_reduce`."""
    pieces = x.unbind(dim)
    out = pieces[0].clone()
    for p in pieces[1:]:
        out = out | p
    return out


def _bb_to_per_sq_bool(bb: torch.Tensor) -> torch.Tensor:
    """(N,) int64 bitboard → (N, 64) bool: True at sq iff bit sq set."""
    bit_idx = torch.arange(64, device=bb.device, dtype=torch.int64)
    return ((bb.unsqueeze(-1) >> bit_idx) & 1).bool()


def _slider_attacks_one_seed(seed_per_sq: torch.Tensor, empty: torch.Tensor,
                             direction: int) -> torch.Tensor:
    """Per-from-sq slider attacks: `seed_per_sq[N, 64]` int64 bitboards
    (one bit per from-sq) → `(N, 64)` int64 attack bitboards in `direction`.

    Vectorized across the 64-source-square axis: a single Kogge-Stone fill
    on `[N, 64]` int64 broadcasts the shared `empty[N, 1]` mask while each
    column carries its own per-square seed. PyTorch's bitwise shift acts
    elementwise on int64, so the same shift instruction handles all 64
    source squares at once. Replaces what used to be a Python `for s in
    range(64)` loop firing 64 separate per-direction GPU launches —
    eliminates ~63 of every 64 launches in the slider-attack hot path.
    """
    g = seed_per_sq                                           # (N, 64)
    e = empty.unsqueeze(-1)                                   # (N, 1) broadcast

    if direction == DIR_N:
        g = g | (e & (g << 8));  e = e & (e << 8)
        g = g | (e & (g << 16)); e = e & (e << 16)
        g = g | (e & (g << 32))
        return g << 8
    if direction == DIR_S:
        g = g | (e & _lsr(g, 8));  e = e & _lsr(e, 8)
        g = g | (e & _lsr(g, 16)); e = e & _lsr(e, 16)
        g = g | (e & _lsr(g, 32))
        return _lsr(g, 8)
    if direction == DIR_E:
        m = NOT_FILE_A
        e = e & m
        g = g | (e & (g << 1)); e = e & (e << 1)
        g = g | (e & (g << 2)); e = e & (e << 2)
        g = g | (e & (g << 4))
        return (g << 1) & m
    if direction == DIR_W:
        m = NOT_FILE_H
        e = e & m
        g = g | (e & _lsr(g, 1)); e = e & _lsr(e, 1)
        g = g | (e & _lsr(g, 2)); e = e & _lsr(e, 2)
        g = g | (e & _lsr(g, 4))
        return _lsr(g, 1) & m
    if direction == DIR_NE:
        m = NOT_FILE_A
        e = e & m
        g = g | (e & (g << 9));  e = e & (e << 9)
        g = g | (e & (g << 18)); e = e & (e << 18)
        g = g | (e & (g << 36))
        return (g << 9) & m
    if direction == DIR_NW:
        m = NOT_FILE_H
        e = e & m
        g = g | (e & (g << 7));  e = e & (e << 7)
        g = g | (e & (g << 14)); e = e & (e << 14)
        g = g | (e & (g << 28))
        return (g << 7) & m
    if direction == DIR_SE:
        m = NOT_FILE_A
        e = e & m
        g = g | (e & _lsr(g, 7));  e = e & _lsr(e, 7)
        g = g | (e & _lsr(g, 14)); e = e & _lsr(e, 14)
        g = g | (e & _lsr(g, 28))
        return _lsr(g, 7) & m
    if direction == DIR_SW:
        m = NOT_FILE_H
        e = e & m
        g = g | (e & _lsr(g, 9));  e = e & _lsr(e, 9)
        g = g | (e & _lsr(g, 18)); e = e & _lsr(e, 18)
        g = g | (e & _lsr(g, 36))
        return _lsr(g, 9) & m
    raise ValueError(f"unknown direction {direction}")


def _bishop_attacks_per_sq(seeds: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(seeds)
    for d in _BISHOP_DIRS:
        out = out | _slider_attacks_one_seed(seeds, empty, d)
    return out


def _rook_attacks_per_sq(seeds: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(seeds)
    for d in _ROOK_DIRS:
        out = out | _slider_attacks_one_seed(seeds, empty, d)
    return out


def _gather_per_game(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """`table[64]` indexed by `idx[N]` → `(N,)`. Used to fetch RAYS[d, king_sq] etc."""
    return table[idx]


def _compute_pawn_pseudo_targets(state: "ChessBatchedState",
                                  our_pawn: torch.Tensor,
                                  our_occ: torch.Tensor,
                                  opp_occ: torch.Tensor,
                                  empty: torch.Tensor) -> torch.Tensor:
    """Per-from-sq legal-target bitboards for our pawns.

    Returns (N, 64) int64. Includes single push, double push (rank 2/7),
    diagonal captures, and en-passant capture target.

    Pawn attacks for capture squares come from `PAWN_ATTACKS_W/B` lookup.
    Pushes come from shifted bitboards. The resulting `(N, 64)` is sparse —
    only nonzero at squares where our pawn sits.
    """
    N = state.n
    device = state.device
    bit_idx = torch.arange(64, device=device, dtype=torch.int64)
    is_white = (state.side == 0)                                  # (N,) bool
    pawn_at_sq = _bb_to_per_sq_bool(our_pawn)                      # (N, 64)

    # Per-square pawn attack bitboard (lookup table; color-dependent).
    attack_w = PAWN_ATTACKS_W.to(device).unsqueeze(0).expand(N, 64)  # (N, 64)
    attack_b = PAWN_ATTACKS_B.to(device).unsqueeze(0).expand(N, 64)
    attack_per_sq = torch.where(is_white.view(N, 1), attack_w, attack_b)  # (N, 64)

    # Capturable squares: opponent occupancy + en-passant target.
    ep_bb = torch.zeros((N,), dtype=torch.int64, device=device)
    ep_valid = state.ep >= 0
    if ep_valid.any():
        # Build EP bb: bit (ep_sq) for valid games.
        ep_idx = state.ep.to(torch.int64).clamp(0, 63)
        # bit shift in int64: (1 << ep_idx). For ep_idx=63 produces sign bit; OK.
        ep_bb = torch.where(ep_valid, torch.tensor(1, dtype=torch.int64, device=device) << ep_idx, ep_bb)
    capture_targets = (opp_occ | ep_bb).unsqueeze(-1).expand(N, 64)  # (N, 64)

    # Pawn capture moves: per-sq attack ∩ capturable, gated by "our pawn here".
    # Without the gate, every square would inherit the pawn-capture bitboard as
    # if a pawn were on it — leaking moves from squares we don't occupy.
    cap_per_sq = torch.where(
        pawn_at_sq, attack_per_sq & capture_targets,
        torch.tensor(0, dtype=torch.int64, device=device),
    )

    # Pawn pushes — built globally per-color, then attributed back to per-from-sq
    # bitboards in a single vectorized pass (no Python `for s in range(64)`).
    one_w_global = (our_pawn << 8) & empty                       # (N,) white single-push targets
    one_b_global = _lsr(our_pawn, 8) & empty                     # (N,) black single-push targets
    # Double push: rank-2 white (sq 8..15) → rank 4; rank-7 black (sq 48..55) → rank 5.
    rank2_mask = int(np.uint64(sum(1 << s for s in range(8, 16))).view(np.int64))
    rank7_mask = int(np.uint64(sum(1 << s for s in range(48, 56))).view(np.int64))
    two_w_global = ((((our_pawn & rank2_mask) << 8) & empty) << 8) & empty
    two_b_global = _lsr(_lsr(our_pawn & rank7_mask, 8) & empty, 8) & empty

    # Vectorized per-from-sq distribution. For each square s the per-from-sq
    # push contribution sets bit (s ± offset) in push_targets[:, s] iff the
    # corresponding global push bb has that target bit set AND our pawn sits
    # at s. Build per-sq target indices + masks once, broadcast against (N,)
    # global bbs.
    sq = torch.arange(64, device=device, dtype=torch.int64)       # (64,)
    one64 = torch.tensor(1, dtype=torch.int64, device=device)
    is_white_b = is_white.view(N, 1)
    is_black_b = ~is_white_b

    # White single push s → s+8 (only valid for s < 56).
    s_plus_8 = (sq + 8).clamp(max=63)                             # (64,)
    valid_w1 = (sq + 8 < 64).unsqueeze(0)                         # (1, 64)
    push_w1_bit = ((one_w_global.unsqueeze(1) >> s_plus_8) & 1).bool()  # (N, 64) target reachable?
    push_w1 = torch.where(
        pawn_at_sq & is_white_b & push_w1_bit & valid_w1,
        one64 << s_plus_8,
        torch.zeros((), dtype=torch.int64, device=device),
    )

    # Black single push s → s-8 (only valid for s >= 8).
    s_minus_8 = (sq - 8).clamp(min=0)
    valid_b1 = (sq >= 8).unsqueeze(0)
    push_b1_bit = ((_lsr_tensor(one_b_global.unsqueeze(1), s_minus_8) & 1).bool()
                   if False else ((one_b_global.unsqueeze(1) >> s_minus_8) & 1).bool())
    # (one_b_global is non-negative per the _lsr definition; right-shift by
    # non-negative s_minus_8 is safe — sign bit was already cleared in
    # _lsr(our_pawn, 8) above.)
    push_b1 = torch.where(
        pawn_at_sq & is_black_b & push_b1_bit & valid_b1,
        one64 << s_minus_8,
        torch.zeros((), dtype=torch.int64, device=device),
    )

    # White double push s → s+16 (only valid for s in 8..15).
    s_plus_16 = (sq + 16).clamp(max=63)
    valid_w2 = ((sq >= 8) & (sq < 16)).unsqueeze(0)
    push_w2_bit = ((two_w_global.unsqueeze(1) >> s_plus_16) & 1).bool()
    push_w2 = torch.where(
        pawn_at_sq & is_white_b & push_w2_bit & valid_w2,
        one64 << s_plus_16,
        torch.zeros((), dtype=torch.int64, device=device),
    )

    # Black double push s → s-16 (only valid for s in 48..55).
    s_minus_16 = (sq - 16).clamp(min=0)
    valid_b2 = ((sq >= 48) & (sq < 56)).unsqueeze(0)
    push_b2_bit = ((two_b_global.unsqueeze(1) >> s_minus_16) & 1).bool()
    push_b2 = torch.where(
        pawn_at_sq & is_black_b & push_b2_bit & valid_b2,
        one64 << s_minus_16,
        torch.zeros((), dtype=torch.int64, device=device),
    )

    push_targets = push_w1 | push_b1 | push_w2 | push_b2
    return cap_per_sq | push_targets


def _compute_pseudo_targets(state: "ChessBatchedState",
                            our_pieces: torch.Tensor, opp_pieces: torch.Tensor,
                            our_occ: torch.Tensor, opp_occ: torch.Tensor,
                            occupancy: torch.Tensor, empty: torch.Tensor,
                            opp_attacks_xray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pseudo_targets[N, 64], pseudo_targets_pawn[N, 64]).

    pseudo_targets is the OR of per-piece-type target bitboards for the piece
    at each from-sq (pawn / knight / bishop / rook / queen / king), filtered
    by `~our_occ` (no friendly captures). King moves are additionally filtered
    by `~opp_attacks_xray` so the king never moves into check.

    pseudo_targets_pawn is just the pawn contribution, kept separate so the
    underpromotion encoding can require a pawn at from_sq.

    No pin/check filter applied yet — see `legal_mask` for those.
    """
    N = our_pieces.shape[0]
    device = our_pieces.device

    pawn_targets = _compute_pawn_pseudo_targets(
        state, our_pieces[:, P_PAWN], our_occ, opp_occ, empty,
    )

    # Per-sq seed bitboards (one-hot per from_sq).
    one = torch.tensor(1, dtype=torch.int64, device=device)
    bit_idx = torch.arange(64, device=device, dtype=torch.int64)
    seeds_per_sq = (one << bit_idx).expand(N, 64).clone()  # (N, 64): bit s set at column s

    # Knight: KNIGHT_ATTACKS[s] for our knight squares, masked by ~our_occ.
    knight_at_sq = _bb_to_per_sq_bool(our_pieces[:, P_KNIGHT])     # (N, 64)
    knight_table = KNIGHT_ATTACKS.to(device).unsqueeze(0).expand(N, 64)
    knight_targets = torch.where(
        knight_at_sq, knight_table & ~our_occ.unsqueeze(-1),
        torch.tensor(0, dtype=torch.int64, device=device),
    )

    # King.
    king_at_sq = _bb_to_per_sq_bool(our_pieces[:, P_KING])
    king_table = KING_ATTACKS.to(device).unsqueeze(0).expand(N, 64)
    king_targets = torch.where(
        king_at_sq,
        king_table & ~our_occ.unsqueeze(-1) & ~opp_attacks_xray.unsqueeze(-1),
        torch.tensor(0, dtype=torch.int64, device=device),
    )

    # Sliders (bishop/rook/queen). Compute per-sq attacks via Kogge-Stone seeded
    # from each square individually. Mask by piece presence + ~our_occ.
    bishop_attacks_per_sq = _bishop_attacks_per_sq(seeds_per_sq, empty)
    rook_attacks_per_sq = _rook_attacks_per_sq(seeds_per_sq, empty)

    bishop_at_sq = _bb_to_per_sq_bool(our_pieces[:, P_BISHOP])
    rook_at_sq = _bb_to_per_sq_bool(our_pieces[:, P_ROOK])
    queen_at_sq = _bb_to_per_sq_bool(our_pieces[:, P_QUEEN])
    not_own = (~our_occ).unsqueeze(-1)

    bishop_t = torch.where(bishop_at_sq, bishop_attacks_per_sq & not_own,
                           torch.tensor(0, dtype=torch.int64, device=device))
    rook_t = torch.where(rook_at_sq, rook_attacks_per_sq & not_own,
                         torch.tensor(0, dtype=torch.int64, device=device))
    queen_t = torch.where(
        queen_at_sq, (bishop_attacks_per_sq | rook_attacks_per_sq) & not_own,
        torch.tensor(0, dtype=torch.int64, device=device),
    )

    pseudo = pawn_targets | knight_targets | king_targets | bishop_t | rook_t | queen_t
    return pseudo, pawn_targets


def _compute_pin_filter(king_bb: torch.Tensor, our_occ: torch.Tensor,
                        opp_rq: torch.Tensor, opp_bq: torch.Tensor,
                        occupancy: torch.Tensor) -> torch.Tensor:
    """Return (N, 64) pin-target filter.

    For each from-sq, the bitboard of squares the piece is allowed to move to
    *given any pin* through the king. If the piece is not pinned, the filter
    is `~0` (all bits set, meaning no restriction). If pinned, the filter is
    the ray through king-and-pinner, including the pinner.
    """
    N = king_bb.shape[0]
    device = king_bb.device
    pin_filter = torch.full((N, 64), -1, dtype=torch.int64, device=device)  # all-bits-set initially
    empty = ~occupancy

    if device.type == "cuda":
        from src.games.chess_gpu_triton import (
            slider_attacks_8way,
            slider_attacks_8way_per_dir_empty,
        )
        # First pass: 8 directions sharing ``empty``.
        first_all = slider_attacks_8way(king_bb, empty)              # (N, 8)
        first_blocker_all = first_all & occupancy.unsqueeze(-1)       # (N, 8)
        # X-ray empties per direction: ~ (occupancy ^ first_blocker[d]).
        # = empty | first_blocker_all[d] (since first_blocker is in occupancy).
        empties_x = empty.unsqueeze(-1) | first_blocker_all            # (N, 8)
        # Second pass: 8 directions with per-direction empty mask.
        second_all = slider_attacks_8way_per_dir_empty(king_bb, empties_x)  # (N, 8)
    else:
        first_all = None

    for d in range(8):
        is_orthogonal = d in (DIR_N, DIR_S, DIR_E, DIR_W)
        opp_slider = opp_rq if is_orthogonal else opp_bq

        if first_all is not None:
            first_attacks = first_all[:, d]                          # (N,)
            first_blocker = first_blocker_all[:, d]                  # (N,)
            second_attacks = second_all[:, d]                        # (N,)
        else:
            first_attacks = _slider_attacks(king_bb, empty, d)
            first_blocker = first_attacks & occupancy
            occ_x = occupancy & ~first_blocker
            empty_x = ~occ_x
            second_attacks = _slider_attacks(king_bb, empty_x, d)

        first_is_ours = (first_blocker & our_occ) != 0               # (N,) bool
        occ_x_d = occupancy & ~first_blocker
        second_blocker = second_attacks & occ_x_d
        second_is_pinner = (second_blocker & opp_slider) != 0        # (N,) bool

        is_pin = first_is_ours & second_is_pinner                    # (N,) bool

        pin_ray = first_attacks | second_attacks                     # (N,)

        pinned_at_sq = _bb_to_per_sq_bool(first_blocker)             # (N, 64) — True at the pinned sq
        pin_filter = torch.where(
            pinned_at_sq & is_pin.unsqueeze(-1),
            pin_ray.unsqueeze(-1),
            pin_filter,
        )

    return pin_filter


def _compute_check_resolve(king_bb: torch.Tensor, our_occ: torch.Tensor,
                           opp_pieces: torch.Tensor, opp_color: int,
                           occupancy: torch.Tensor, empty: torch.Tensor,
                           is_white: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute check-resolution bitboard + double-check flag.

    Returns:
        resolve: (N,) int64 bb. When in single-check, non-king moves must
            land on a square in this bb (capture-the-checker or block).
            When not in check, all-bits-set (no restriction).
            When in double-check, zero (only king moves allowed).
        double_check: (N,) bool.
    """
    N = king_bb.shape[0]
    device = king_bb.device

    # Pieces attacking our king square (= "checkers").
    # Compute by treating our king as if it were each opponent piece type and
    # checking which opp pieces it'd hit. Symmetry: pawn attacks from king's
    # perspective use opposite-color pawn lookup.
    bit_idx = torch.arange(64, device=device, dtype=torch.int64)
    # King square as the "from" for symmetric attack.
    # Pawn checkers: opp pawns on squares from which they attack king. Equivalent:
    # a pseudo-pawn of OUR color on king's square (its diag attacks point to where
    # an opp pawn would be sitting to check). Use our color's pawn attack table.
    pawn_table_us = PAWN_ATTACKS_W.to(device) if True else PAWN_ATTACKS_B.to(device)
    # For per-game color choice:
    pawn_table_w = PAWN_ATTACKS_W.to(device)
    pawn_table_b = PAWN_ATTACKS_B.to(device)
    # per-game pawn-attack-from-king-sq using our color's table
    # (so it returns squares where opp pawns would be to check us).
    king_pawn_attacks_w = _from_sq_attacks(king_bb, pawn_table_w)  # used when we are white
    king_pawn_attacks_b = _from_sq_attacks(king_bb, pawn_table_b)
    king_pawn_attacks = torch.where(is_white, king_pawn_attacks_w, king_pawn_attacks_b)

    pawn_checkers = king_pawn_attacks & opp_pieces[:, P_PAWN]
    knight_checkers = _from_sq_attacks(king_bb, KNIGHT_ATTACKS.to(device)) & opp_pieces[:, P_KNIGHT]

    # Slider checkers: cast bishop/rook attacks from king_bb through actual
    # occupancy; intersect with opp sliders. Use the fused 8-direction
    # Triton kernel — one launch covers both diagonals and orthogonals.
    if king_bb.device.type == "cuda":
        from src.games.chess_gpu_triton import slider_attacks_8way
        king_attacks = slider_attacks_8way(king_bb, empty)             # (N, 8)
        bishop_attacks_from_king = (
            king_attacks[:, DIR_NE] | king_attacks[:, DIR_SE]
            | king_attacks[:, DIR_SW] | king_attacks[:, DIR_NW]
        )
        rook_attacks_from_king = (
            king_attacks[:, DIR_N] | king_attacks[:, DIR_S]
            | king_attacks[:, DIR_E] | king_attacks[:, DIR_W]
        )
    else:
        bishop_attacks_from_king = (
            _slider_attacks(king_bb, empty, DIR_NE) | _slider_attacks(king_bb, empty, DIR_NW)
            | _slider_attacks(king_bb, empty, DIR_SE) | _slider_attacks(king_bb, empty, DIR_SW)
        )
        rook_attacks_from_king = (
            _slider_attacks(king_bb, empty, DIR_N) | _slider_attacks(king_bb, empty, DIR_S)
            | _slider_attacks(king_bb, empty, DIR_E) | _slider_attacks(king_bb, empty, DIR_W)
        )
    bishop_checkers = bishop_attacks_from_king & (opp_pieces[:, P_BISHOP] | opp_pieces[:, P_QUEEN])
    rook_checkers = rook_attacks_from_king & (opp_pieces[:, P_ROOK] | opp_pieces[:, P_QUEEN])

    checkers = pawn_checkers | knight_checkers | bishop_checkers | rook_checkers
    in_check = checkers != 0

    # Popcount via repeated shift-and-mask.
    n_checkers = _popcount(checkers)
    double_check = n_checkers >= 2

    # Single-check resolve: capture checker | block (between king and slider checker).
    # For non-slider checkers (pawn/knight), only capture is possible.
    # For slider checkers, also block on between(king, checker).
    slider_checker = bishop_checkers | rook_checkers
    # `between` bitboard: squares strictly between king and slider checker.
    # Compute by combining king's slider attacks toward the checker with the checker's
    # slider attacks toward the king — their intersection is the between-squares.
    # Easier: use the first_attacks for the slider direction. Already implicit in
    # `_slider_attacks(king_bb, empty, d)` for the direction containing the checker,
    # which gives king→checker (incl. checker). The "between" is that minus the
    # checker bit.
    between_bb = torch.zeros_like(king_bb)
    for d in range(8):
        ray_attacks = _slider_attacks(king_bb, empty, d)
        # Mask in the slider-checker that lies in this direction.
        checker_in_dir = ray_attacks & slider_checker  # (N,) — single-bit if checker in this direction
        # The squares between king and the checker = ray_attacks ^ checker_in_dir
        # (only if checker_in_dir is set; else 0).
        between_in_dir = torch.where(
            checker_in_dir != 0, ray_attacks ^ checker_in_dir,
            torch.tensor(0, dtype=torch.int64, device=device),
        )
        between_bb = between_bb | between_in_dir

    resolve_single = checkers | between_bb  # (N,) — bb of squares non-king moves must land on
    # When double-check: only king moves allowed — set resolve to 0.
    resolve_single = torch.where(double_check, torch.tensor(0, dtype=torch.int64, device=device), resolve_single)
    # When not in check: full mask.
    resolve = torch.where(in_check, resolve_single, torch.tensor(-1, dtype=torch.int64, device=device))
    return resolve, in_check, double_check, checkers


def _popcount(bb: torch.Tensor) -> torch.Tensor:
    """Bit popcount on int64. SWAR-style; vectorized over batch."""
    # Standard SWAR popcount adapted for int64.
    m1 = 0x5555555555555555
    m2 = 0x3333333333333333
    m4 = 0x0f0f0f0f0f0f0f0f
    h01 = 0x0101010101010101
    # Convert to unsigned-like by using _lsr for arithmetic-safe right-shift.
    x = bb - ((_lsr(bb, 1)) & m1)
    x = (x & m2) + (_lsr(x, 2) & m2)
    x = (x + _lsr(x, 4)) & m4
    return _lsr(x * h01, 56)


def _add_castling_targets(king_targets: torch.Tensor, state: "ChessBatchedState",
                          our_occ: torch.Tensor, opp_occ: torch.Tensor,
                          opp_attacks: torch.Tensor, in_check: torch.Tensor,
                          is_white: torch.Tensor) -> torch.Tensor:
    """Augment per-from-sq king targets with legal castling moves.

    Adds king-target squares (g1/c1 for white, g8/c8 for black) when castling
    rights are intact, the path between king and rook is empty, the squares
    the king passes through (incl. start and dest) aren't attacked, and we're
    not currently in check.

    `king_targets[N, king_sq]` is the only entry that gets mutated.
    """
    device = king_targets.device
    occupancy = our_occ | opp_occ
    one = torch.tensor(1, dtype=torch.int64, device=device)

    # Convenience squares.
    E1, F1, G1, D1, C1, B1 = 4, 5, 6, 3, 2, 1
    E8, F8, G8, D8, C8, B8 = 60, 61, 62, 59, 58, 57

    # Squares that must be empty between king and rook for each castle type.
    wk_path_empty = (one << F1) | (one << G1)
    wq_path_empty = (one << D1) | (one << C1) | (one << B1)
    bk_path_empty = (one << F8) | (one << G8)
    bq_path_empty = (one << D8) | (one << C8) | (one << B8)

    # Squares that must NOT be attacked (king's path: e1→f1→g1 for WK, etc.).
    wk_safe = (one << E1) | (one << F1) | (one << G1)
    wq_safe = (one << E1) | (one << D1) | (one << C1)
    bk_safe = (one << E8) | (one << F8) | (one << G8)
    bq_safe = (one << E8) | (one << D8) | (one << C8)

    rights = state.castling  # (N, 4) bool — [WK, WQ, BK, BQ]

    def _try_castle(eligible: torch.Tensor, path_empty: int, safe: int,
                    king_sq: int, target_sq: int) -> torch.Tensor:
        """For each game where `eligible`, if path is empty and safe is unattacked,
        add bit(target_sq) to king_targets[N, king_sq]."""
        path_clear = (occupancy & path_empty) == 0
        safe_clear = (opp_attacks & safe) == 0
        ok = eligible & path_clear & safe_clear & ~in_check
        return torch.where(
            ok,
            king_targets[:, king_sq] | (one << target_sq),
            king_targets[:, king_sq],
        )

    king_targets = king_targets.clone()
    king_targets[:, E1] = _try_castle(rights[:, CR_WK] & is_white, wk_path_empty, wk_safe, E1, G1)
    king_targets[:, E1] = _try_castle(rights[:, CR_WQ] & is_white, wq_path_empty, wq_safe, E1, C1)
    king_targets[:, E8] = _try_castle(rights[:, CR_BK] & ~is_white, bk_path_empty, bk_safe, E8, G8)
    king_targets[:, E8] = _try_castle(rights[:, CR_BQ] & ~is_white, bq_path_empty, bq_safe, E8, C8)
    return king_targets


def _legal_mask_impl(state: "ChessBatchedState") -> tuple[torch.Tensor, torch.Tensor]:
    """Internal entry point. See GpuChessGame.legal_mask for docstring."""
    N = state.n
    device = state.device
    side = state.side.to(torch.int64)                              # (N,) 0 or 1
    is_white = (side == 0)                                          # (N,) bool

    # Reorganize pieces as (N, 2, 6) → gather along color axis.
    pieces_2x6 = state.pieces.view(N, 2, 6)
    us_idx = side.view(N, 1, 1).expand(N, 1, 6)                    # (N, 1, 6)
    our_pieces = pieces_2x6.gather(1, us_idx).squeeze(1)           # (N, 6)
    opp_pieces = pieces_2x6.gather(1, 1 - us_idx).squeeze(1)       # (N, 6)

    our_occ = _or_reduce(our_pieces, dim=-1)                        # (N,)
    opp_occ = _or_reduce(opp_pieces, dim=-1)                        # (N,)
    occupancy = our_occ | opp_occ
    empty = ~occupancy

    our_king_bb = our_pieces[:, P_KING]                             # (N,)

    # Opponent attacks with our king masked out (king-x-ray): used for king
    # legality and castling-path safety.
    occ_no_king = occupancy & ~our_king_bb
    attacks_w = attacks_by_color(state, 0, occ_no_king)
    attacks_b = attacks_by_color(state, 1, occ_no_king)
    opp_attacks_xray = torch.where(is_white, attacks_b, attacks_w)  # (N,)

    # Pseudo-legal targets per from-sq.
    pseudo, pseudo_pawn = _compute_pseudo_targets(
        state, our_pieces, opp_pieces, our_occ, opp_occ, occupancy, empty,
        opp_attacks_xray,
    )

    # Castling: augment king's target bb.
    pseudo_with_castle = _add_castling_targets(
        pseudo, state, our_occ, opp_occ, opp_attacks_xray,
        in_check=torch.tensor([False] * N, device=device),  # placeholder; recomputed below
        is_white=is_white,
    )

    # Pin filter.
    opp_rq = opp_pieces[:, P_ROOK] | opp_pieces[:, P_QUEEN]
    opp_bq = opp_pieces[:, P_BISHOP] | opp_pieces[:, P_QUEEN]
    pin_filter = _compute_pin_filter(our_king_bb, our_occ, opp_rq, opp_bq, occupancy)

    # Check resolve mask + check status.
    resolve, in_check, double_check, _checkers = _compute_check_resolve(
        our_king_bb, our_occ, opp_pieces, 1 - 0, occupancy, empty, is_white,
    )

    # Re-do castling now that we know in_check (the placeholder above didn't
    # filter it out; do so here properly).
    pseudo = _add_castling_targets(
        pseudo, state, our_occ, opp_occ, opp_attacks_xray, in_check, is_white,
    )

    # Apply pin filter (no-op at king's slot — pin_filter is -1 there).
    pseudo = pseudo & pin_filter
    pseudo_pawn = pseudo_pawn & pin_filter

    # Apply check-resolve mask to non-king squares only. King moves are
    # already filtered by `~opp_attacks_xray` inside _compute_pseudo_targets.
    king_at_sq = _bb_to_per_sq_bool(our_king_bb)                    # (N, 64)
    pseudo = torch.where(king_at_sq, pseudo, pseudo & resolve.unsqueeze(-1))
    # pseudo_pawn never has king moves; resolve applies uniformly.
    pseudo_pawn = pseudo_pawn & resolve.unsqueeze(-1)

    # EP-resolves-check edge case: when the side to move is in check by a pawn
    # that just double-pushed (= the EP-captured pawn IS the checker), EP
    # capture resolves the check — but the move's TO-square (the EP target,
    # one rank past the captured pawn) isn't in the standard resolve mask, so
    # the filter above incorrectly strips it. Re-add the EP-capture target
    # specifically for pawns that can reach it diagonally.
    one = torch.tensor(1, dtype=torch.int64, device=device)
    zero = torch.tensor(0, dtype=torch.int64, device=device)
    ep_valid = state.ep >= 0
    ep_idx_safe = state.ep.to(torch.int64).clamp(0, 63)
    # Captured-pawn square is one rank "behind" the EP target relative to mover
    # (white captures north → captured pawn is south of target).
    ep_captured_sq = torch.where(is_white, ep_idx_safe - 8, ep_idx_safe + 8).clamp(0, 63)
    ep_captured_bb = torch.where(ep_valid, one << ep_captured_sq, zero)
    ep_target_bb = torch.where(ep_valid, one << ep_idx_safe, zero)
    _, in_check_now, _, checkers_now = _compute_check_resolve(
        our_king_bb, our_occ, opp_pieces, 1 - 0, occupancy, empty, is_white,
    )
    ep_resolves = in_check_now & ((checkers_now & ep_captured_bb) != 0) & ep_valid

    # Per-from-sq EP-capture contribution: for pawns whose diagonal attack
    # bitboard includes the EP target. `pin_filter` still applies (a pinned
    # pawn that can't capture along its pin ray won't gain the EP move).
    our_pawn = our_pieces[:, P_PAWN]
    pawn_at_sq = _bb_to_per_sq_bool(our_pawn)                        # (N, 64)
    attack_w = PAWN_ATTACKS_W.to(device).unsqueeze(0).expand(N, 64)
    attack_b = PAWN_ATTACKS_B.to(device).unsqueeze(0).expand(N, 64)
    attack_per_sq = torch.where(is_white.view(N, 1), attack_w, attack_b)
    ep_capture_per_sq = attack_per_sq & ep_target_bb.view(N, 1)      # (N, 64)
    ep_capture_per_sq = torch.where(pawn_at_sq, ep_capture_per_sq, zero)
    ep_capture_per_sq = ep_capture_per_sq & pin_filter

    add_back = ep_resolves.view(N, 1)
    pseudo = torch.where(add_back, pseudo | ep_capture_per_sq, pseudo)
    pseudo_pawn = torch.where(add_back, pseudo_pawn | ep_capture_per_sq, pseudo_pawn)

    # EP-pin (horizontal): when our king sits on the same rank as the EP-
    # captured pawn AND we have a pawn adjacent to that captured pawn, the EP
    # capture removes BOTH pawns from the rank in one move. If an opp rook/
    # queen sits on that rank with no other piece between it and the king
    # (after both pawns are gone), the king is exposed. Standard pin filter
    # misses this because the pin is created BY the EP move, not pre-existing.
    king_sq = king_at_sq.long().argmax(dim=-1)                      # (N,)
    king_rank = king_sq // 8
    ep_captured_rank = ep_captured_sq // 8
    king_on_ep_rank = (king_rank == ep_captured_rank) & ep_valid     # (N,)

    for delta in (-1, 1):
        cand_sq = ep_captured_sq + delta                             # (N,)
        cand_valid = (
            ep_valid
            & (cand_sq >= 0) & (cand_sq < 64)
            & ((cand_sq // 8) == ep_captured_rank)
        )
        cand_sq_safe = cand_sq.clamp(0, 63)
        cand_bb = torch.where(cand_valid, one << cand_sq_safe, zero)
        has_pawn = (our_pawn & cand_bb) != 0

        # Simulate removing both the candidate-pawn and the ep-captured pawn.
        occ_x = occupancy & ~cand_bb & ~ep_captured_bb
        empty_x = ~occ_x
        e_attacks_x = _slider_attacks(our_king_bb, empty_x, DIR_E)
        w_attacks_x = _slider_attacks(our_king_bb, empty_x, DIR_W)
        rank_attacks_x = e_attacks_x | w_attacks_x
        hits_opp_rq_x = (rank_attacks_x & opp_rq) != 0

        illegal = king_on_ep_rank & has_pawn & hits_opp_rq_x         # (N,)

        cand_at_sq = _bb_to_per_sq_bool(cand_bb)                      # (N, 64)
        illegal_at_sq = cand_at_sq & illegal.view(N, 1)               # (N, 64)
        pseudo = torch.where(
            illegal_at_sq, pseudo & ~ep_target_bb.view(N, 1), pseudo,
        )
        pseudo_pawn = torch.where(
            illegal_at_sq, pseudo_pawn & ~ep_target_bb.view(N, 1), pseudo_pawn,
        )

    # Encode pseudo_targets into action mask (N, 4672).
    # Actions are STM-encoded: from_sq and to_sq are always "as if white".
    # Convert STM → absolute before indexing into pseudo_targets (absolute coords).
    target_stm = ACTION_TARGET_W.to(device)                         # (4672,) — STM to_sq
    is_underpromo = ACTION_IS_UNDERPROMO.to(device)
    from_sq_stm = ACTION_FROM_SQ.to(device)                         # (4672,) — STM from_sq

    from_sq_stm_exp = from_sq_stm.unsqueeze(0).expand(N, 4672)
    target_stm_exp = target_stm.unsqueeze(0).expand(N, 4672)
    is_w = is_white.unsqueeze(-1)

    # STM → absolute: for black, rank-flip via ^ 56.
    from_sq_abs = torch.where(is_w, from_sq_stm_exp, from_sq_stm_exp ^ 56)
    target_abs = torch.where(is_w, target_stm_exp, target_stm_exp ^ 56)
    action_valid = target_stm_exp >= 0                               # (N, 4672) bool
    target_abs = torch.where(action_valid, target_abs,
                             torch.tensor(-1, dtype=torch.int64, device=device))

    # For each (game, action_idx): bit `target_abs` of pseudo[game, from_sq_abs].
    pseudo_at_from = pseudo.gather(1, from_sq_abs)                   # (N, 4672) int64
    bit_at_target = (_lsr_tensor(pseudo_at_from, target_abs.clamp(min=0)) & 1).bool()
    legal_general = bit_at_target & action_valid

    # For underpromo actions: additionally require pawn at from_sq on the
    # STM promotion rank. In STM coords, promotion is always rank 6 → 7
    # (mover's pawns always go "north").
    pseudo_pawn_at_from = pseudo_pawn.gather(1, from_sq_abs)
    bit_at_target_pawn = (_lsr_tensor(pseudo_pawn_at_from, target_abs.clamp(min=0)) & 1).bool()
    from_rank_stm = from_sq_stm // 8                                 # (4672,)
    promo_rank_ok = (from_rank_stm == 6).unsqueeze(0).expand(N, 4672)
    legal_pawn_specific = bit_at_target_pawn & action_valid & promo_rank_ok

    # Combine: underpromo idxs use pawn-specific check; others use general.
    legal_mask = torch.where(is_underpromo.unsqueeze(0), legal_pawn_specific, legal_general)

    return legal_mask, in_check


def _lsr_tensor(g: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Logical right shift where `n` is a per-element int64 tensor.

    `(g >> n) & ((1 << (64 - n)) - 1)` — but `n` varies per element. Compute
    the mask elementwise via `(1 << (64 - n)) - 1`. For `n == 0` the mask is
    `(1 << 64) - 1` which doesn't fit in int64; clamp `n` to ≥ 1 if needed
    (caller guarantees n ≥ 1 for valid action targets, but n=0 is also fine
    because then no shift is needed).
    """
    # For a per-element n in [0, 63], compute `(g >> n) & ~(((-1) << (64 - n))`
    # equivalently. Simpler: treat n=0 specially (no shift needed).
    # Use the identity: (g >> n) & ((1 << (64-n)) - 1).
    # For batched n, we evaluate (1 << (64-n)) - 1 elementwise via tensor ops.
    one = torch.tensor(1, dtype=torch.int64, device=g.device)
    # (64 - n) ranges over [1, 64]; for n=0 the mask is (1 << 64) - 1 → not representable as int64.
    # Use a different identity for n=0: result = g (no shift, no mask change needed besides bit count truncation, but bits are already in range).
    n_safe = n.clamp(min=1, max=63)
    mask = (one << (64 - n_safe)) - 1
    shifted = g >> n_safe
    masked = shifted & mask
    # When n == 0, return g itself. But typically callers pass n >= 0; n == 0 means
    # we want bit 0 of the full bitboard. (g >> 0) & 1 = bit 0 of g. So the n=0 case
    # would ideally produce: bit 0 of g. With our path (n_safe=1): we'd shift by 1
    # and mask by ~bit 63 — giving bit 1 of g. That's WRONG for n=0.
    # Fix: select between shifted+masked and unshifted-g via where(n == 0).
    return torch.where(n == 0, g, masked)


# -- batched state -----------------------------------------------------------
@dataclass
class ChessBatchedState(BatchedGameState):
    """Bitboard chess state for N games.

    All tensors share device + leading dim N. Integer dtypes are sized
    conservatively so an N×state footprint stays small even at large N.
    """

    pieces: torch.Tensor    # (N, 12) int64. Order: W{P,N,B,R,Q,K} then B{P,N,B,R,Q,K}.
    side: torch.Tensor      # (N,) int8 — 0 = white to move, 1 = black to move.
    castling: torch.Tensor  # (N, 4) bool — [WK, WQ, BK, BQ].
    ep: torch.Tensor        # (N,) int16 — ep target square 0..63 or -1.
    halfmove: torch.Tensor  # (N,) int16 — 50-move clock.
    fullmove: torch.Tensor  # (N,) int16 — fullmove number (starts at 1).
    ply: torch.Tensor       # (N,) int16 — total plies elapsed.
    done: torch.Tensor      # (N,) bool.
    winner: torch.Tensor    # (N,) int8 — +1, 0, -1; only meaningful when done.

    # Threefold-repetition history (Phase 3). The buffer holds Zobrist hashes
    # of the positions reached since the last "irreversibility" — capture,
    # pawn move, castling-rights change, or EP-availability change. Two
    # positions are repeated only if both match on side/castling/EP, which is
    # baked into the hash. K=160 is sized for the 75-move auto-draw cap
    # (halfmove ≥ 150) with a small margin; irreversibility resets here cover
    # a superset of the events that reset the halfmove clock.
    rep_hashes: torch.Tensor | None = None   # (N, REP_HISTORY_K) int64.
    rep_count: torch.Tensor | None = None    # (N,) int16 — # valid entries.

    @property
    def n(self) -> int:
        return self.pieces.shape[0]

    @property
    def device(self) -> torch.device:
        return self.pieces.device


# -- game ---------------------------------------------------------------------
class GpuChessGame(BatchedGame):
    board_size = (8, 8)
    action_space_size = ACTION_SPACE
    num_planes = NUM_PLANES
    max_plies = 1024  # mirror ChessGame.max_plies — same draw-cutoff semantics. Bumped 400→1024 (2026-05-08) — see ChessGame.max_plies comment for rationale.

    def reset_batch(
        self, n: int, *, device: str | torch.device = "cpu",
    ) -> ChessBatchedState:
        return self.from_python_chess([chess.Board() for _ in range(n)], device=device)

    def from_python_chess(
        self,
        boards: list[chess.Board],
        *,
        device: str | torch.device = "cpu",
    ) -> ChessBatchedState:
        """Build a batched state from a list of python-chess `Board` objects.

        Used for testing (oracle parity) and bootstrapping. Construction is on
        CPU then `.to(device)` at the end — the loop is Python-bound anyway.
        """
        n = len(boards)
        # Build bitboards in uint64 then view as int64 — torch lacks broad uint64
        # support; the bit-extract `(bb >> i) & 1` is sign-safe because the `& 1`
        # mask strips any sign-extended high bits from the arithmetic shift.
        pieces_u64 = np.zeros((n, 12), dtype=np.uint64)
        side_np = np.zeros((n,), dtype=np.int8)
        castling_np = np.zeros((n, 4), dtype=bool)
        ep_np = np.full((n,), -1, dtype=np.int16)
        halfmove_np = np.zeros((n,), dtype=np.int16)
        fullmove_np = np.zeros((n,), dtype=np.int16)
        ply_np = np.zeros((n,), dtype=np.int16)

        for i, board in enumerate(boards):
            for color, color_offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
                for piece_type, plane_idx in _PIECE_TYPE_TO_IDX.items():
                    pieces_u64[i, color_offset + plane_idx] = int(board.pieces(piece_type, color))
            side_np[i] = 0 if board.turn == chess.WHITE else 1
            castling_np[i, CR_WK] = board.has_kingside_castling_rights(chess.WHITE)
            castling_np[i, CR_WQ] = board.has_queenside_castling_rights(chess.WHITE)
            castling_np[i, CR_BK] = board.has_kingside_castling_rights(chess.BLACK)
            castling_np[i, CR_BQ] = board.has_queenside_castling_rights(chess.BLACK)
            ep_np[i] = -1 if board.ep_square is None else board.ep_square
            halfmove_np[i] = board.halfmove_clock
            fullmove_np[i] = board.fullmove_number
            ply_np[i] = board.ply()

        state = ChessBatchedState(
            pieces=torch.from_numpy(pieces_u64.view(np.int64).copy()).to(device),
            side=torch.from_numpy(side_np).to(device),
            castling=torch.from_numpy(castling_np).to(device),
            ep=torch.from_numpy(ep_np).to(device),
            halfmove=torch.from_numpy(halfmove_np).to(device),
            fullmove=torch.from_numpy(fullmove_np).to(device),
            ply=torch.from_numpy(ply_np).to(device),
            done=torch.zeros((n,), dtype=torch.bool, device=device),
            winner=torch.zeros((n,), dtype=torch.int8, device=device),
        )
        # Initialize rep_hashes ring with the starting Zobrist hash.
        h = _zobrist_hash(state)
        rep_hashes = torch.zeros((n, REP_HISTORY_K), dtype=torch.int64, device=device)
        rep_hashes[:, 0] = h
        rep_count = torch.ones((n,), dtype=torch.int16, device=device)
        state.rep_hashes = rep_hashes
        state.rep_count = rep_count
        return state

    def to_tensor_batch(self, state: ChessBatchedState) -> torch.Tensor:
        """Return `(N, 19, 8, 8)` observation tensor in STM-relative encoding.

        Plane layout (must match `ChessGame.to_tensor`):
          0..5   : own P, N, B, R, Q, K (side-to-move's pieces)
          6..11  : opponent P, N, B, R, Q, K
          12..13 : own castling rights KS, QS (broadcast)
          14..15 : opponent castling rights KS, QS (broadcast)
          16     : en-passant target square (one-hot, zeros if no ep)
          17     : turn (1.0 white-to-move, 0.0 black-to-move) (broadcast)
          18     : fullmove_number / 200 clipped to 1.0 (broadcast)

        For black-to-move: ranks are flipped (row 0 = black's back rank),
        own/opponent planes are swapped, castling planes reordered.
        """
        n = state.n
        device = state.device
        is_black = (state.side == 1)

        # Piece planes via bit-extract.
        bit_idx = torch.arange(64, device=device, dtype=torch.int64)
        bb = state.pieces.unsqueeze(-1)                                      # (N, 12, 1)
        bits = ((bb >> bit_idx) & 1).to(torch.float32)                       # (N, 12, 64)
        piece_planes = bits.view(n, 12, 8, 8)                                # row=sq//8, col=sq%8

        # STM: for black, flip ranks and swap own(white)/opp(black) planes.
        if is_black.any():
            flipped = piece_planes.flip(2)
            swapped = torch.cat([flipped[:, 6:12], flipped[:, 0:6]], dim=1)
            piece_planes = torch.where(
                is_black.view(n, 1, 1, 1), swapped, piece_planes,
            )

        # Castling: STM reorders [WK,WQ,BK,BQ] → [own_KS,own_QS,opp_KS,opp_QS].
        castle_abs = state.castling.to(torch.float32)                        # (N, 4)
        castle_stm = torch.where(
            is_black.view(n, 1),
            torch.cat([castle_abs[:, 2:4], castle_abs[:, 0:2]], dim=1),
            castle_abs,
        )
        castle_planes = castle_stm.view(n, 4, 1, 1).expand(n, 4, 8, 8)

        # EP plane: for black, flip square rank (sq ^ 56).
        ep_plane = torch.zeros((n, 1, 8, 8), dtype=torch.float32, device=device)
        ep_valid = state.ep >= 0
        ep_idx = state.ep.to(torch.int64).clamp(0, 63)
        ep_idx_stm = torch.where(is_black, ep_idx ^ 56, ep_idx)
        ep_row = ep_idx_stm // 8
        ep_col = ep_idx_stm % 8
        ep_plane[torch.arange(n, device=device), 0, ep_row, ep_col] = ep_valid.to(torch.float32)

        # Turn plane: 1 if white to move (side == 0), else 0.
        turn_plane = (~is_black).to(torch.float32).view(n, 1, 1, 1).expand(n, 1, 8, 8)

        # Move-count plane: clipped fullmove / 200.
        mc = (state.fullmove.to(torch.float32) / 200.0).clamp(0.0, 1.0)
        mc_plane = mc.view(n, 1, 1, 1).expand(n, 1, 8, 8)

        return torch.cat(
            [piece_planes, castle_planes, ep_plane, turn_plane, mc_plane], dim=1
        )

    def legal_mask(self, state: ChessBatchedState) -> torch.Tensor:
        """Return `(N, 4672)` bool mask of legal actions per game.

        Pin-aware, check-aware, with castling / en-passant / promotions
        encoded to match `chess.py:_move_to_action`. Cross-validated against
        `ChessGame.legal_actions` on a corpus of random positions
        (see `tests/test_chess_gpu_legal.py`).
        """
        legal, _in_check = _legal_mask_impl(state)
        return legal

    def step_batch(
        self,
        state: ChessBatchedState,
        actions: torch.Tensor,
    ) -> tuple[ChessBatchedState, torch.Tensor, torch.Tensor]:
        """Apply one action per game; return (new_state, reward, done).

        Reward is mover-POV (+1 for delivering checkmate, 0 otherwise) to
        match `Game.step`'s convention. Terminals: mate / stalemate /
        50-move / threefold (Zobrist-hash) / ply cap (`max_plies`).
        """
        new_state, reward, done, _ = _step_batch_impl(state, actions, self.max_plies)
        return new_state, reward, done

    def step_batch_with_legal(
        self,
        state: ChessBatchedState,
        actions: torch.Tensor,
    ) -> tuple[ChessBatchedState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like ``step_batch`` but also returns the new state's legal-action
        mask. ``_step_batch_impl`` already computes it internally for
        terminal detection — exposing it here lets callers (self-play
        loop) skip a duplicate ~16 ms ``legal_mask`` call at the next ply.
        """
        return _step_batch_impl(state, actions, self.max_plies)


# -- Zobrist hashing for threefold detection ---------------------------------
REP_HISTORY_K = 160  # ring buffer depth; see ChessBatchedState.rep_hashes docstring.

# Random int64s seeded once at import. The rng has a fixed seed so hashes are
# deterministic across runs (useful for comparing buffers / replays).
_zob_rng = np.random.default_rng(0xC0DEC0DE)
_ZOBRIST_PIECE_NP = _zob_rng.integers(0, 2**64, (12, 64), dtype=np.uint64).view(np.int64).copy()
_ZOBRIST_SIDE_NP = int(np.int64(_zob_rng.integers(0, 2**64, dtype=np.uint64).view(np.int64)))
_ZOBRIST_CASTLING_NP = _zob_rng.integers(0, 2**64, (16,), dtype=np.uint64).view(np.int64).copy()
_ZOBRIST_EP_NP = _zob_rng.integers(0, 2**64, (8,), dtype=np.uint64).view(np.int64).copy()

ZOBRIST_PIECE = torch.from_numpy(_ZOBRIST_PIECE_NP)
ZOBRIST_SIDE = _ZOBRIST_SIDE_NP
ZOBRIST_CASTLING = torch.from_numpy(_ZOBRIST_CASTLING_NP)
ZOBRIST_EP = torch.from_numpy(_ZOBRIST_EP_NP)


def _zobrist_hash(state: ChessBatchedState) -> torch.Tensor:
    """Compute (N,) int64 Zobrist hash of the batched state.

    Full re-hash each call (no incremental update). 12 × 64 = 768 batched
    bit-tests + a handful of small lookups. Tolerable per-step cost; can
    be made incremental in a follow-up if profiling shows it matters.
    """
    N = state.n
    device = state.device
    bit_idx = torch.arange(64, device=device, dtype=torch.int64)
    # (N, 12, 64) bool: bit `sq` of pieces[g, pt] set?
    bits = ((state.pieces.unsqueeze(-1) >> bit_idx) & 1).bool()
    z_piece = ZOBRIST_PIECE.to(device)
    contributions = torch.where(
        bits, z_piece.unsqueeze(0),
        torch.tensor(0, dtype=torch.int64, device=device),
    )                                                                # (N, 12, 64)
    # XOR-reduce via pairwise halving — torch has no native bitwise_xor reduce,
    # but pairwise XOR is associative so log2(768) levels suffice.
    flat = contributions.view(N, 12 * 64)
    while flat.shape[-1] > 1:
        if flat.shape[-1] % 2 == 1:
            pad = torch.zeros((N, 1), dtype=torch.int64, device=device)
            flat = torch.cat([flat, pad], dim=-1)
        flat = flat[:, 0::2] ^ flat[:, 1::2]
    h = flat.squeeze(-1)
    h = torch.where(state.side == 1, h ^ ZOBRIST_SIDE, h)
    # Castling rights: 4 bits → index 0..15.
    rights = state.castling.to(torch.int64)
    cr_idx = rights[:, 0] * 8 + rights[:, 1] * 4 + rights[:, 2] * 2 + rights[:, 3]
    z_castling = ZOBRIST_CASTLING.to(device)
    h = h ^ z_castling[cr_idx]
    # EP file (0..7), only when ep set.
    has_ep = state.ep >= 0
    ep_file = state.ep.to(torch.int64).clamp(0, 63) % 8
    z_ep = ZOBRIST_EP.to(device)
    h = torch.where(has_ep, h ^ z_ep[ep_file], h)
    return h


# -- step_batch implementation -----------------------------------------------
# Castling-rook squares per (color, side).
_W_KS_RFROM, _W_KS_RTO = 7, 5     # h1 → f1
_W_QS_RFROM, _W_QS_RTO = 0, 3     # a1 → d1
_B_KS_RFROM, _B_KS_RTO = 63, 61   # h8 → f8
_B_QS_RFROM, _B_QS_RTO = 56, 59   # a8 → d8

# Underpromo piece-id (chess.py 0=R/1=B/2=N) → our piece-type index.
_UNDERPROMO_TO_PT = (P_ROOK, P_BISHOP, P_KNIGHT)


def _insufficient_material(state: "ChessBatchedState") -> torch.Tensor:
    """Batched python-chess-exact ``board.is_insufficient_material()``.

    Mirrors ``chess.Board.has_insufficient_material(color)`` for both colors
    and ANDs them (bug_hunt_2026_06_13.md §C — the GPU engine was missing the
    insufficient-material draw term, flooding the buffer with artificial draws).

    Returns a (N,) bool: True iff NEITHER side has winning material, by the
    exact material/bishop-square-color rules python-chess uses:
      - a side with any pawn/rook/queen is sufficient;
      - a side holding a knight is insufficient only if it has nothing else
        (popcount(side) <= 2, i.e. king + that one knight) AND the opponent has
        no non-king/non-queen pieces (no selfmate material);
      - a side holding a bishop is insufficient only if ALL bishops on the board
        (both colors) are on one square color AND there are no pawns and no
        knights anywhere;
      - a bare king (king only) is always insufficient.

    This is an EXACT match, not a conservative subset: it returns True on
    exactly the set python-chess does. Verified by cross-validation in
    tests/test_chess_gpu_insufficient_material.py (0 false positives AND 0
    false negatives across constructed + >=1000 random positions).
    """
    device = state.device
    p = state.pieces  # (N, 12) int64 bitboards; white base 0, black base 6.

    def _occ(base: int) -> torch.Tensor:
        occ = p[:, base + P_PAWN]
        for off in (P_KNIGHT, P_BISHOP, P_ROOK, P_QUEEN, P_KING):
            occ = occ | p[:, base + off]
        return occ

    w_occ = _occ(0)
    b_occ = _occ(6)

    pawns = p[:, P_PAWN] | p[:, 6 + P_PAWN]
    knights = p[:, P_KNIGHT] | p[:, 6 + P_KNIGHT]
    bishops = p[:, P_BISHOP] | p[:, 6 + P_BISHOP]
    rooks = p[:, P_ROOK] | p[:, 6 + P_ROOK]
    queens = p[:, P_QUEEN] | p[:, 6 + P_QUEEN]

    dark = torch.tensor(BB_DARK_SQUARES, dtype=torch.int64, device=device)
    light = torch.tensor(BB_LIGHT_SQUARES, dtype=torch.int64, device=device)
    zero = torch.zeros((), dtype=torch.int64, device=device)

    # All bishops (both colors) confined to a single square color.
    no_dark_bishops = (bishops & dark) == zero
    no_light_bishops = (bishops & light) == zero
    bishops_same_color = no_dark_bishops | no_light_bishops
    no_pawns = pawns == zero
    no_knights = knights == zero

    def _has_insufficient(base: int, occ: torch.Tensor,
                          opp_occ: torch.Tensor) -> torch.Tensor:
        # (1) any own pawn/rook/queen → sufficient.
        own_prq = (occ & (pawns | rooks | queens)) != zero

        # Knight branch: own knight present.
        has_knight = (occ & knights) != zero
        # opponent's non-king, non-queen material (pawns/knights/bishops/rooks).
        kings_bb = p[:, P_KING] | p[:, 6 + P_KING]
        opp_selfmate = (opp_occ & ~kings_bb & ~queens) != zero
        knight_insuff = (_popcount(occ) <= 2) & (~opp_selfmate)

        # Bishop branch: own bishop present (and no own knight, since the
        # knight branch returns first in python-chess).
        has_bishop = (occ & bishops) != zero
        bishop_insuff = bishops_same_color & no_pawns & no_knights

        # python-chess control flow (first matching branch wins):
        #   if own_prq: False
        #   elif has_knight: knight_insuff
        #   elif has_bishop: bishop_insuff
        #   else: True   (bare king)
        result = torch.where(
            own_prq,
            torch.zeros_like(own_prq),
            torch.where(
                has_knight,
                knight_insuff,
                torch.where(has_bishop, bishop_insuff,
                            torch.ones_like(own_prq)),
            ),
        )
        return result

    w_insuff = _has_insufficient(0, w_occ, b_occ)
    b_insuff = _has_insufficient(6, b_occ, w_occ)
    return w_insuff & b_insuff


def _step_batch_impl(
    state: ChessBatchedState,
    actions: torch.Tensor,
    max_plies: int,
) -> tuple[ChessBatchedState, torch.Tensor, torch.Tensor]:
    N = state.n
    device = state.device
    side = state.side.to(torch.int64)                              # (N,)
    is_white = side == 0                                            # (N,) bool

    # Lookup action metadata. Actions are STM-encoded (as-if-white);
    # convert from_sq and to_sq to absolute coordinates for internal use.
    target_stm_all = ACTION_TARGET_W.to(device)                      # STM table
    is_underpromo_all = ACTION_IS_UNDERPROMO.to(device)
    promo_piece_all = _ACTION_TABLES["promo_piece"].to(device)
    from_sq_stm_all = ACTION_FROM_SQ.to(device)

    actions_long = actions.to(torch.int64)
    from_sq_stm = from_sq_stm_all[actions_long]                     # (N,) STM from_sq
    to_sq_stm = target_stm_all[actions_long]                         # (N,) STM to_sq
    # STM → absolute: for black, rank-flip via ^ 56.
    from_sq = torch.where(is_white, from_sq_stm, from_sq_stm ^ 56)
    action_target = torch.where(is_white, to_sq_stm, to_sq_stm ^ 56)
    is_underpromo = is_underpromo_all[actions_long]                  # (N,) bool
    promo_piece = promo_piece_all[actions_long]                      # (N,) int64; -1 if not underpromo

    one = torch.tensor(1, dtype=torch.int64, device=device)
    zero = torch.tensor(0, dtype=torch.int64, device=device)

    from_bb = one << from_sq.clamp(0, 63)                            # (N,)
    to_bb = one << action_target.clamp(0, 63)                        # (N,)

    # Reorganize pieces.
    pieces_2x6 = state.pieces.view(N, 2, 6)
    us_idx = side.view(N, 1, 1).expand(N, 1, 6)
    our = pieces_2x6.gather(1, us_idx).squeeze(1).clone()             # (N, 6)
    opp = pieces_2x6.gather(1, 1 - us_idx).squeeze(1).clone()         # (N, 6)

    # Identify our piece type at from_sq.
    our_at_from = (our & from_bb.unsqueeze(1)) != 0                  # (N, 6) bool

    # Promotion bookkeeping.
    target_rank = action_target // 8
    is_pawn_move = our_at_from[:, P_PAWN]
    pawn_to_back_rank = is_pawn_move & (
        (is_white & (target_rank == 7)) | (~is_white & (target_rank == 0))
    )                                                                 # (N,)

    # Underpromo target piece type (only meaningful when is_underpromo).
    promo_pt_table = torch.tensor(_UNDERPROMO_TO_PT, dtype=torch.int64, device=device)
    promo_pt_idx = torch.where(is_underpromo, promo_pt_table[promo_piece.clamp(0, 2)],
                               torch.tensor(P_QUEEN, dtype=torch.int64, device=device))
    is_promotion = pawn_to_back_rank
    # Note: pawn-to-back-rank via queen-type action → queen promo; via underpromo
    # action → R/B/N. Both are captured by `is_promotion`; piece type chosen by
    # `promo_pt_idx` (defaulting to queen for queen-type to back rank).

    # Build per-piece-type remove and add masks for our side.
    our_remove = torch.zeros((N, 6), dtype=torch.int64, device=device)
    our_add = torch.zeros((N, 6), dtype=torch.int64, device=device)
    for pt in range(6):
        # Always remove from_bb if our piece pt was at from_sq (the moving piece leaves).
        our_remove[:, pt] = torch.where(our_at_from[:, pt], from_bb, zero)
        # Add to_bb if pt is the destination piece type:
        #   - promotion: pt == promo_pt_idx
        #   - non-promotion: our piece pt was at from_sq (same type lands at to_sq)
        is_dest = torch.where(
            is_promotion,
            (promo_pt_idx == pt) & is_pawn_move,
            our_at_from[:, pt],
        )
        our_add[:, pt] = torch.where(is_dest, to_bb, zero)

    # Castling: also move the rook.
    is_king_move = our_at_from[:, P_KING]
    is_castle_ks = is_king_move & ((action_target - from_sq) == 2)
    is_castle_qs = is_king_move & ((action_target - from_sq) == -2)
    is_castling = is_castle_ks | is_castle_qs

    rook_from_sq = (
        torch.where(is_castle_ks & is_white, torch.tensor(_W_KS_RFROM, dtype=torch.int64, device=device),
        torch.where(is_castle_qs & is_white, torch.tensor(_W_QS_RFROM, dtype=torch.int64, device=device),
        torch.where(is_castle_ks & ~is_white, torch.tensor(_B_KS_RFROM, dtype=torch.int64, device=device),
        torch.where(is_castle_qs & ~is_white, torch.tensor(_B_QS_RFROM, dtype=torch.int64, device=device),
                    zero))))
    )
    rook_to_sq = (
        torch.where(is_castle_ks & is_white, torch.tensor(_W_KS_RTO, dtype=torch.int64, device=device),
        torch.where(is_castle_qs & is_white, torch.tensor(_W_QS_RTO, dtype=torch.int64, device=device),
        torch.where(is_castle_ks & ~is_white, torch.tensor(_B_KS_RTO, dtype=torch.int64, device=device),
        torch.where(is_castle_qs & ~is_white, torch.tensor(_B_QS_RTO, dtype=torch.int64, device=device),
                    zero))))
    )
    rook_from_bb = torch.where(is_castling, one << rook_from_sq, zero)
    rook_to_bb = torch.where(is_castling, one << rook_to_sq, zero)
    our_remove[:, P_ROOK] = our_remove[:, P_ROOK] | rook_from_bb
    our_add[:, P_ROOK] = our_add[:, P_ROOK] | rook_to_bb

    # Capture: clear opp piece at to_sq (any piece type).
    opp_at_to = (opp & to_bb.unsqueeze(1)) != 0                       # (N, 6) bool
    is_normal_capture = opp_at_to.any(dim=1)                          # (N,)
    opp_remove = torch.where(opp_at_to, to_bb.unsqueeze(1), zero)     # (N, 6)

    # En-passant capture: pawn move where to_bb == ep target.
    ep_valid = state.ep >= 0
    ep_target_bb = torch.where(ep_valid, one << state.ep.to(torch.int64).clamp(0, 63), zero)
    is_ep_capture = is_pawn_move & ep_valid & (to_bb == ep_target_bb)
    ep_captured_sq = torch.where(is_white, action_target - 8, action_target + 8).clamp(0, 63)
    ep_captured_bb = torch.where(is_ep_capture, one << ep_captured_sq, zero)
    opp_remove[:, P_PAWN] = opp_remove[:, P_PAWN] | ep_captured_bb

    is_capture = is_normal_capture | is_ep_capture

    # Apply piece updates.
    new_our = (our & ~our_remove) | our_add
    new_opp = opp & ~opp_remove

    # Scatter back into (N, 12). Compose new pieces tensor without mutating the input.
    new_pieces_2x6 = pieces_2x6.clone()
    new_pieces_2x6.scatter_(1, us_idx, new_our.unsqueeze(1))
    new_pieces_2x6.scatter_(1, 1 - us_idx, new_opp.unsqueeze(1))
    new_pieces = new_pieces_2x6.view(N, 12)

    # Castling rights update.
    new_castling = state.castling.clone()
    F = torch.tensor(False, device=device)

    def _clear_if(cond: torch.Tensor, slot: int) -> None:
        new_castling[:, slot] = torch.where(cond, F, new_castling[:, slot])

    _clear_if(is_king_move & is_white, CR_WK)
    _clear_if(is_king_move & is_white, CR_WQ)
    _clear_if(is_king_move & ~is_white, CR_BK)
    _clear_if(is_king_move & ~is_white, CR_BQ)

    rook_moved = our_at_from[:, P_ROOK]
    _clear_if(rook_moved & is_white & (from_sq == 7), CR_WK)
    _clear_if(rook_moved & is_white & (from_sq == 0), CR_WQ)
    _clear_if(rook_moved & ~is_white & (from_sq == 63), CR_BK)
    _clear_if(rook_moved & ~is_white & (from_sq == 56), CR_BQ)

    captured_opp_rook = opp_at_to[:, P_ROOK]
    _clear_if(captured_opp_rook & ~is_white & (action_target == 7), CR_WK)
    _clear_if(captured_opp_rook & ~is_white & (action_target == 0), CR_WQ)
    _clear_if(captured_opp_rook & is_white & (action_target == 63), CR_BK)
    _clear_if(captured_opp_rook & is_white & (action_target == 56), CR_BQ)

    # New EP target: set on a double pawn push only.
    double_push = is_pawn_move & ((action_target - from_sq).abs() == 16)
    new_ep = torch.where(double_push, (from_sq + action_target) // 2,
                         torch.tensor(-1, dtype=torch.int64, device=device)).to(torch.int16)

    # Halfmove clock.
    new_halfmove = torch.where(
        is_capture | is_pawn_move,
        torch.tensor(0, dtype=torch.int64, device=device),
        state.halfmove.to(torch.int64) + 1,
    ).to(torch.int16)

    # Fullmove (+1 only after black moves).
    new_fullmove = torch.where(~is_white,
                               state.fullmove.to(torch.int64) + 1,
                               state.fullmove.to(torch.int64)).to(torch.int16)

    new_side = (1 - state.side.to(torch.int64)).to(torch.int8)
    new_ply = (state.ply.to(torch.int64) + 1).to(torch.int16)

    # Freeze ALREADY-done games (state.done True BEFORE this step): carry every
    # board field forward unchanged so a finished game is fully static under
    # later sentinel steps. Without this the board keeps mutating and the
    # end-of-batch terminal observation (self_play.py final_obs) is corrupt for
    # every game that finished before the batch's longest. Games that become
    # newly-done THIS step are unaffected (their gate is state.done == False).
    # See bug_hunt_2026_06_13.md §D.
    already_done = state.done
    d1 = already_done.view(N, 1)  # broadcast over (N, *) per-field shapes
    new_pieces = torch.where(d1, state.pieces, new_pieces)
    new_side = torch.where(already_done, state.side, new_side)
    new_castling = torch.where(d1, state.castling, new_castling)
    new_ep = torch.where(already_done, state.ep, new_ep)
    new_halfmove = torch.where(already_done, state.halfmove, new_halfmove)
    new_fullmove = torch.where(already_done, state.fullmove, new_fullmove)
    new_ply = torch.where(already_done, state.ply, new_ply)

    # Provisional new state (terminals computed below).
    new_state = ChessBatchedState(
        pieces=new_pieces,
        side=new_side,
        castling=new_castling,
        ep=new_ep,
        halfmove=new_halfmove,
        fullmove=new_fullmove,
        ply=new_ply,
        done=state.done.clone(),
        winner=state.winner.clone(),
    )

    # Compute Zobrist hash of new state.
    new_hash = _zobrist_hash(new_state)

    # Update rep_hashes ring (initialize lazily if state didn't have one).
    if state.rep_hashes is None:
        old_rep_hashes = torch.zeros((N, REP_HISTORY_K), dtype=torch.int64, device=device)
        old_rep_count = torch.zeros((N,), dtype=torch.int16, device=device)
    else:
        old_rep_hashes = state.rep_hashes
        old_rep_count = state.rep_count

    # Irreversibility for repetition tracking is broader than for halfmove:
    # rep tracking also resets on castling-rights changes and EP-target
    # changes (since two positions are repetitions only if both match).
    rights_old = state.castling.to(torch.int64)
    rights_new = new_castling.to(torch.int64)
    cr_idx_old = rights_old[:, 0] * 8 + rights_old[:, 1] * 4 + rights_old[:, 2] * 2 + rights_old[:, 3]
    cr_idx_new = rights_new[:, 0] * 8 + rights_new[:, 1] * 4 + rights_new[:, 2] * 2 + rights_new[:, 3]
    castling_changed = cr_idx_old != cr_idx_new
    ep_changed = (state.ep != new_ep)
    irreversible = is_capture | is_pawn_move | castling_changed | ep_changed | is_castling

    new_rep_hashes = old_rep_hashes.clone()
    rep_idx_safe = torch.where(
        irreversible,
        torch.tensor(0, dtype=torch.int64, device=device),
        old_rep_count.to(torch.int64).clamp(0, REP_HISTORY_K - 1),
    )
    new_rep_hashes.scatter_(1, rep_idx_safe.unsqueeze(1), new_hash.unsqueeze(1))
    new_rep_count = torch.where(
        irreversible,
        torch.tensor(1, dtype=torch.int64, device=device),
        (old_rep_count.to(torch.int64) + 1).clamp(max=REP_HISTORY_K),
    ).to(torch.int16)

    # Threefold detection: count occurrences of new_hash in valid segment.
    rep_pos = torch.arange(REP_HISTORY_K, device=device).unsqueeze(0).expand(N, REP_HISTORY_K)
    valid = rep_pos < new_rep_count.to(torch.int64).unsqueeze(1)
    matches = (new_rep_hashes == new_hash.unsqueeze(1)) & valid
    threefold = matches.sum(dim=1) >= 3

    # Freeze the repetition ring for already-done games too (§D). A done game's
    # board is static, so its rep history must not absorb sentinel-step hashes.
    new_rep_hashes = torch.where(d1, old_rep_hashes, new_rep_hashes)
    new_rep_count = torch.where(already_done, old_rep_count, new_rep_count)

    new_state.rep_hashes = new_rep_hashes
    new_state.rep_count = new_rep_count

    # Terminal detection: compute legal_mask of new_state. The legal-mask
    # impl computes ``in_check`` internally (via _compute_check_resolve),
    # so we get it for free — saves a duplicate ~8 ms attacks_by_color × 2
    # that the previous version did separately.
    new_legal, in_check_new = _legal_mask_impl(new_state)
    no_legal = new_legal.sum(dim=-1) == 0

    mate = no_legal & in_check_new
    stalemate = no_legal & ~in_check_new
    # 75-move rule (auto) — not the claimable 50-move rule. Matches
    # python-chess `is_seventyfive_moves` (halfmove_clock >= 150) which
    # ChessGame.step picks up via `is_game_over(claim_draw=False)`.
    seventy_five_move = new_halfmove >= 150
    ply_cap = new_ply >= max_plies
    # Insufficient-material draw, matching python-chess
    # is_insufficient_material() (bug_hunt_2026_06_13.md §C). Winner stays 0
    # (a draw, like the other draw terminals below).
    insufficient_material = _insufficient_material(new_state)

    done = mate | stalemate | seventy_five_move | ply_cap | threefold | insufficient_material

    # Mover-POV reward (matches Game.step). Mover = side just moved = old side.
    # Already-done games emit reward 0 — no terminal reward is re-emitted under
    # sentinel steps (§D).
    reward = torch.where(mate & ~already_done, 1.0, 0.0).to(torch.float32)

    # Winner from white's POV: +1 if white wins, -1 if black wins, 0 draw.
    winner = torch.where(
        mate,
        torch.where(is_white, torch.tensor(1, dtype=torch.int8, device=device),
                              torch.tensor(-1, dtype=torch.int8, device=device)),
        torch.tensor(0, dtype=torch.int8, device=device),
    )

    new_state.done = state.done | done  # sticky once a game is done
    new_state.winner = torch.where(state.done, state.winner, winner)

    # Return new_legal alongside — step_batch's terminal detection just
    # computed it, and the caller (self-play loop) needs it again at the
    # next ply for MCTS root masking. Returning it here avoids a duplicate
    # ~16 ms _legal_mask_impl call per ply.
    return new_state, reward, done, new_legal
