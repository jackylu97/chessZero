"""D4 board-symmetry augmentation for pawnless, castle-free training samples.

Chess positions without pawns or castling rights are invariant under the full
8-element dihedral group (4 rotations × mirror) — every piece's movement rules
survive the transforms. Applying a RANDOM group element to each eligible
sampled training window (observations + actions + policy targets + legal
masks, all under the SAME element; value/DTM/outcome are invariants) makes
absolute-square memorization impossible: the only hypothesis consistent with
the data is relative geometry (king-to-edge distance, opposition, the box) —
the substrate of general endgame technique. KataGo trains with Go's 8-fold
symmetry the same way (random transform per sample, storage unchanged).

Soundness rests on the true game symmetry
    transform(step(s, a)) == step(transform(s), transform(a))
which tests/test_symmetry.py verifies against python-chess move generation
(NOT against these tables — the legality of mapped moves on independently
rule-checked transformed boards breaks any circularity).

Conventions: everything operates in the MOVER frame (the frame observations
and actions are already encoded in — `_move_to_action` collapses black to the
white frame internally), with the grid indexed [rank_row, file_col] as
produced by `ChessGame.to_tensor` (row = sq // 8, col = sq % 8). Group
elements are parameterized g = flip^f ∘ rot90^k (k in 0..3, f in 0..1);
g == 0 is the identity. Square and move-type tables are derived from the SAME
torch grid ops applied to an index grid, so observation and action transforms
are consistent by construction.
"""
from __future__ import annotations

import numpy as np
import torch

from src.games.chess import NUM_MOVE_TYPES  # 73

N_SYM = 8
_A = 64 * NUM_MOVE_TYPES

# Codec displacement tables (mirrors _action_to_move exactly).
_DIR_MAP = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1),
            4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1)}
_KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def _grid_op(x: torch.Tensor, g: int) -> torch.Tensor:
    """Apply group element g to the last two dims (the [row, col] board grid)."""
    k, f = g % 4, g // 4
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if f:
        x = torch.flip(x, dims=(-1,))
    return x


def _build_tables():
    sq_maps = np.zeros((N_SYM, 64), dtype=np.int64)
    vec_maps = []  # per g: (dr, dc) -> (dr', dc') linear map, as a 2x2 matrix
    base = torch.arange(64).view(8, 8)
    for g in range(N_SYM):
        tg = _grid_op(base, g)
        # Piece on old square s sits at new grid position (r, c) where
        # tg[r, c] == s; its new square index is r*8 + c.
        pos = {int(tg[r, c]): r * 8 + c for r in range(8) for c in range(8)}
        for s in range(64):
            sq_maps[g, s] = pos[s]
        # Linear part on displacement vectors, derived from two mapped squares.
        def _rc(s):
            return divmod(int(s), 8)
        p0, p1r, p1c = _rc(sq_maps[g, 0]), _rc(sq_maps[g, 8]), _rc(sq_maps[g, 1])
        # old +(1,0) row step maps to (p1r - p0); old +(0,1) col step -> (p1c - p0)
        m = np.array([[p1r[0] - p0[0], p1c[0] - p0[0]],
                      [p1r[1] - p0[1], p1c[1] - p0[1]]], dtype=np.int64)
        vec_maps.append(m)

    # Move-type displacement table from the codec's own definitions.
    mt_disp = {}
    for mt in range(64):
        if mt < 56:
            dr, dc = _DIR_MAP[mt // 7]
            dist = mt % 7 + 1
            mt_disp[mt] = (dr * dist, dc * dist)
        else:
            mt_disp[mt] = _KNIGHT[mt - 56]
    disp_to_mt = {v: k for k, v in mt_disp.items()}

    mt_maps = np.zeros((N_SYM, NUM_MOVE_TYPES), dtype=np.int64)
    for g in range(N_SYM):
        m = vec_maps[g]
        for mt in range(NUM_MOVE_TYPES):
            if mt >= 64:
                # Underpromotions: pawns present => window ineligible; identity
                # keeps the permutation total.
                mt_maps[g, mt] = mt
                continue
            dr, dc = mt_disp[mt]
            dr2 = int(m[0, 0] * dr + m[0, 1] * dc)
            dc2 = int(m[1, 0] * dr + m[1, 1] * dc)
            mt_maps[g, mt] = disp_to_mt[(dr2, dc2)]

    act_maps = np.zeros((N_SYM, _A), dtype=np.int64)
    for g in range(N_SYM):
        for s in range(64):
            base_new = sq_maps[g, s] * NUM_MOVE_TYPES
            for mt in range(NUM_MOVE_TYPES):
                act_maps[g, s * NUM_MOVE_TYPES + mt] = base_new + mt_maps[g, mt]
    return sq_maps, act_maps


SQ_MAPS, ACT_MAPS = _build_tables()


def transform_obs(obs: torch.Tensor, g: int) -> torch.Tensor:
    """[..., H, W] observation (single frame or stacked history) under g.
    Uniform scalar planes (castling/turn/counters) are invariant under grid
    ops, so applying the transform to every plane is safe."""
    if g == 0:
        return obs
    return _grid_op(obs, g).contiguous()


def transform_actions(actions, g: int):
    """Map an int array/list of action indices under g."""
    if g == 0:
        return actions
    amap = ACT_MAPS[g]
    if isinstance(actions, np.ndarray):
        return amap[actions]
    return [int(amap[a]) for a in actions]


def transform_policy_dense(policy: np.ndarray, g: int) -> np.ndarray:
    """Dense [A] policy target under g: mass on action a moves to g(a)."""
    if g == 0:
        return policy
    out = np.zeros_like(policy)
    out[ACT_MAPS[g]] = policy
    return out


def transform_legal_mask(mask, g: int):
    """Dense [A] bool/float legal mask under g."""
    if g == 0:
        return mask
    if isinstance(mask, torch.Tensor):
        out = torch.zeros_like(mask)
        out[torch.from_numpy(ACT_MAPS[g]).to(mask.device)] = mask
        return out
    out = np.zeros_like(mask)
    out[ACT_MAPS[g]] = mask
    return out


def window_eligible(obs_list, num_planes: int = 22) -> bool:
    """A K-step window may be transformed iff every frame of every stacked
    observation shows no pawns (planes 0, 6) and no castling rights (12-15).
    History-stacked obs have layout [T*C, H, W]; zero-padded past frames are
    trivially eligible."""
    for obs in obs_list:
        c_total = obs.shape[0]
        for off in range(0, c_total, num_planes):
            pawns = obs[off + 0].sum() + obs[off + 6].sum()
            castle = obs[off + 12:off + 16].sum()
            if float(pawns) != 0.0 or float(castle) != 0.0:
                return False
    return True
