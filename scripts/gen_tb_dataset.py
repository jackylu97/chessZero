"""Generate a supervised dataset of random legal <=5-man positions labeled with
GROUND-TRUTH tablebase value (WDL) + optimal-move policy. No self-play, no relabel —
pure Syzygy supervision. Saves (fen, value_class, optimal_action_indices) tuples.

value_class: 0=win, 1=draw, 2=loss  (side-to-move POV, 50-move-rule aware)
optimal policy: uniform over the moves that preserve the best achievable outcome
  (win → min-DTZ winning moves; loss → max-DTZ longest defense; draw → draw-preservers)

Run: PYTHONPATH=. .venv/bin/python scripts/gen_tb_dataset.py --n 300000 --out data/tb5_train.pkl
"""
import argparse, pickle, random, math, os
import chess, chess.syzygy
import multiprocessing as mp
from src.games.chess import _move_to_action

SYZYGY = "data/syzygy"
PT = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
_TB = None


def _init(path):
    global _TB
    _TB = chess.syzygy.open_tablebase(path)


def _random_board(rng):
    for _ in range(60):
        n_extra = rng.randint(1, 3)
        sqs = rng.sample(range(64), 2 + n_extra)
        b = chess.Board.empty()
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        ok = True
        for i in range(n_extra):
            pt = rng.choice(PT); col = rng.choice([chess.WHITE, chess.BLACK]); sq = sqs[2 + i]
            if pt == chess.PAWN and (sq < 8 or sq >= 56):
                pt = chess.QUEEN  # no pawns on rank 1/8
            b.set_piece_at(sq, chess.Piece(pt, col))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        b.clear_stack()
        if b.is_valid() and not b.is_checkmate() and not b.is_stalemate():
            return b
    return None


def _optimal_policy(b):
    """(list[action_idx], value_class) or None. value_class 0=win/1=draw/2=loss STM POV."""
    moves = list(b.legal_moves)
    if not moves:
        return None
    rows = []  # (move, mover_outcome in {1,0,-1}, dtz)
    for m in moves:
        b.push(m)
        try:
            if b.is_checkmate():
                rows.append((m, 1, 0))
            elif b.is_stalemate() or b.is_insufficient_material():
                rows.append((m, 0, 0))
            else:
                cw = _TB.probe_wdl(b)          # child STM (opponent) POV
                mv = -cw                         # mover POV
                cls = 1 if mv >= 2 else (-1 if mv <= -2 else 0)  # 50-move: cursed/blessed -> draw
                dtz = abs(int(_TB.probe_dtz(b))) if cls != 0 else 0
                rows.append((m, cls, dtz))
        except (KeyError, ValueError, chess.syzygy.MissingTableError):
            pass
        finally:
            b.pop()
    if not rows:
        return None
    best = max(c for _, c, _ in rows)
    if best == 1:        # winning: fastest (min DTZ)
        cand = [(m, d) for m, c, d in rows if c == 1]; dm = min(d for _, d in cand)
        opt = [m for m, d in cand if d == dm]
    elif best == -1:     # losing: longest defense (max DTZ)
        cand = [(m, d) for m, c, d in rows if c == -1]; dm = max(d for _, d in cand)
        opt = [m for m, d in cand if d == dm]
    else:                # drawing
        opt = [m for m, c, d in rows if c == 0]
    vcls = {1: 0, 0: 1, -1: 2}[best]
    acts = [_move_to_action(m, b.turn) for m in opt]
    return acts, vcls


def _worker(arg):
    n, seed = arg
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        b = _random_board(rng)
        if b is None:
            continue
        try:
            _TB.probe_wdl(b)
        except Exception:
            continue
        res = _optimal_policy(b)
        if res is None:
            continue
        acts, vcls = res
        out.append((b.fen(), vcls, acts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    per = math.ceil(a.n / a.workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(SYZYGY,)) as pool:
        chunks = pool.map(_worker, [(per, 1000 + i) for i in range(a.workers)])
    data = [x for c in chunks for x in c][:a.n]
    from collections import Counter
    dist = Counter(v for _, v, _ in data)
    pickle.dump(data, open(a.out, "wb"))
    print(f"saved {len(data)} positions to {a.out}")
    print(f"value dist: win={dist[0]} draw={dist[1]} loss={dist[2]} "
          f"({100*dist[0]/len(data):.0f}/{100*dist[1]/len(data):.0f}/{100*dist[2]/len(data):.0f}%)")
    print(f"mean optimal-move-set size: {sum(len(a_) for _,_,a_ in data)/len(data):.2f}")


if __name__ == "__main__":
    main()
