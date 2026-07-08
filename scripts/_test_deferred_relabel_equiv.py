"""Equivalence gate for the deferred TB relabel.

Proves the post-game FEN-pure relabel path (SyzygyRootProber.relabel_position and the
pooled relabel_fens) produces byte-for-byte the SAME value / DTM / policy targets the
inline root_move_values produces for the same boards. If this passes, deferral changes
only WHEN the probes run, not their result.

Run: PYTHONPATH=. .venv/bin/python scripts/_test_deferred_relabel_equiv.py
"""
import sys, math, random
import torch, chess

from src.games.chess_gpu import GpuChessGame
from src.games.syzygy_probe import SyzygyRootProber, state_to_board, relabel_fens, _prober_params

SYZYGY = "data/syzygy"
GAVIOTA = "data/gaviota"
SEEDS = "data/endgame_seeds.txt"
N = 400
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _pol_eq(a, b):
    """Compare two soft policies (idx[], w[]) or None for equality."""
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    ia, wa = a; ib, wb = b
    ia = list(map(int, ia)); ib = list(map(int, ib))
    da = dict(zip(ia, map(float, wa))); db = dict(zip(ib, map(float, wb)))
    if set(da) != set(db):
        return False
    return all(abs(da[k] - db[k]) < 1e-6 for k in da)


def _scalar_eq(x, y):
    xn, yn = (x != x), (y != y)   # NaN check
    if xn or yn:
        return xn and yn
    return abs(float(x) - float(y)) < 1e-6


def main():
    random.seed(0)
    fens = [ln.strip() for ln in open(SEEDS) if ln.strip()]
    random.shuffle(fens)
    fens = fens[:N]
    boards = [chess.Board(f) for f in fens]

    game = GpuChessGame()
    state = game.from_python_chess(boards, device=DEV)
    legal = game.legal_mask(state)

    prober = SyzygyRootProber(
        SYZYGY, max_pieces=5, dtz_weight=0.05, draw_score=0.0,
        value_dtz_shape=0.5, gaviota_path=GAVIOTA,
        policy_win_thresh=0.5, policy_temp=0.15)

    # INLINE path (ground truth).
    prober.root_move_values(state, legal, per_move=True)
    in_val = prober.last_position_value.cpu().tolist()
    in_ml = prober.last_position_moves_left.cpu().tolist()
    in_pol = prober.last_position_policy
    in_tb = prober.in_tb_mask(state).cpu().tolist()

    # DEFERRED single-position path — reconstruct the SAME board via FEN round-trip.
    nbad = 0
    n_checked = 0
    for i in range(len(boards)):
        if not in_tb[i]:
            continue
        fen = state_to_board(state, i).fen()
        pv, ml, pol = prober.relabel_position(chess.Board(fen), want_policy=True)
        n_checked += 1
        ok = (_scalar_eq(pv, in_val[i]) and _scalar_eq(ml, in_ml[i])
              and _pol_eq(pol, in_pol[i] if in_pol is not None else None))
        if not ok:
            nbad += 1
            if nbad <= 5:
                print(f"  MISMATCH i={i} fen={fen}")
                print(f"    val {pv} vs {in_val[i]} | ml {ml} vs {in_ml[i]}")
                print(f"    pol {pol}\n        vs {in_pol[i] if in_pol else None}")
    print(f"single-process deferred: {n_checked} in-TB positions, {nbad} mismatches")

    # POOLED path — same FENs, 4 spawn workers; compare to inline.
    params = _prober_params(prober, SYZYGY, GAVIOTA)
    all_fens = [state_to_board(state, i).fen() for i in range(len(boards)) if in_tb[i]]
    fenmap = relabel_fens(all_fens, params, workers=4, want_policy=True)
    pbad = 0
    j = 0
    for i in range(len(boards)):
        if not in_tb[i]:
            continue
        fen = state_to_board(state, i).fen()
        pv, ml, pol = fenmap[fen]
        ok = (_scalar_eq(pv, in_val[i]) and _scalar_eq(ml, in_ml[i])
              and _pol_eq(pol, in_pol[i] if in_pol is not None else None))
        if not ok:
            pbad += 1
            if pbad <= 5:
                print(f"  POOL MISMATCH i={i} fen={fen}: {pv}/{ml}/{pol}")
        j += 1
    print(f"pooled (4 workers):      {j} in-TB positions, {pbad} mismatches")

    ok = (nbad == 0 and pbad == 0 and n_checked > 50)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
