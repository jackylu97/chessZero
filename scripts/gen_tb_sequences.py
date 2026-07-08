"""Generate K-step SEQUENCES from random <=5-man positions, labeled with ground-truth TB
value + optimal-move policy at every ply — so the DYNAMICS net can be trained via the
K-step unroll, exactly like warmstart games. Actions are a mix of optimal (60%) and random
legal (40%) so the dynamics learns both the optimal-line transitions MCTS exploits AND the
diverse transitions it explores.

Run: PYTHONPATH=. .venv/bin/python scripts/gen_tb_sequences.py --n 200000 --k 5 --out data/tb5_seq.pkl
"""
import argparse, pickle, random, math
import chess, chess.syzygy
import multiprocessing as mp
from src.games.chess import _move_to_action

SYZYGY = "data/syzygy"
PT = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
DRAW_SENTINEL = 150.0   # moves-left for drawn/no-mate positions: "far" (clamps to support edge)
_TB = None
def _init(p):
    global _TB; _TB = chess.syzygy.open_tablebase(p)

def _rand_board(rng):
    for _ in range(60):
        n = rng.randint(1, 3); sq = rng.sample(range(64), 2 + n)
        b = chess.Board.empty()
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        for i in range(n):
            pt = rng.choice(PT); s = sq[2+i]
            if pt == chess.PAWN and (s < 8 or s >= 56): pt = chess.QUEEN
            b.set_piece_at(s, chess.Piece(pt, rng.choice([chess.WHITE, chess.BLACK])))
        b.turn = rng.choice([chess.WHITE, chess.BLACK]); b.clear_stack()
        if b.is_valid() and not b.is_checkmate() and not b.is_stalemate():
            return b
    return None

def _label(b):
    """(value_class 0win/1draw/2loss, [optimal_action_idx]) or None."""
    moves = list(b.legal_moves)
    if not moves: return None
    rows = []
    for m in moves:
        b.push(m)
        try:
            if b.is_checkmate(): rows.append((m, 1, 0))
            elif b.is_stalemate() or b.is_insufficient_material(): rows.append((m, 0, 0))
            else:
                mv = -_TB.probe_wdl(b)
                cls = 1 if mv >= 2 else (-1 if mv <= -2 else 0)
                dtz = abs(int(_TB.probe_dtz(b))) if cls != 0 else 0
                rows.append((m, cls, dtz))
        except Exception: pass
        finally: b.pop()
    if not rows: return None
    best = max(c for _, c, _ in rows)
    if best == 1:
        cand = [(m, d) for m, c, d in rows if c == 1]; dm = min(d for _, d in cand); opt = [m for m, d in cand if d == dm]
    elif best == -1:
        cand = [(m, d) for m, c, d in rows if c == -1]; dm = max(d for _, d in cand); opt = [m for m, d in cand if d == dm]
    else:
        opt = [m for m, c, d in rows if c == 0]
    vcls = {1: 0, 0: 1, -1: 2}[best]
    # moves-left (DTM-style progress target): |DTZ| plies-to-zeroing for decisive
    # positions (smaller = closer to converting), a "far" sentinel for draws so a
    # move that converts a win into a draw makes moves-left jump UP -> search avoids it.
    if vcls == 1:
        ml = DRAW_SENTINEL
    else:
        try:
            ml = float(min(abs(int(_TB.probe_dtz(b))), 200))
        except Exception:
            ml = DRAW_SENTINEL
    return vcls, [_move_to_action(m, b.turn) for m in opt], opt, ml

def _worker(arg):
    n, k, seed = arg; rng = random.Random(seed); out = []; tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        b = _rand_board(rng)
        if b is None: continue
        try: _TB.probe_wdl(b)
        except Exception: continue
        fens = []; acts = []; vcs = []; pols = []; mls = []
        ok = True
        for step in range(k + 1):
            lab = _label(b)
            if lab is None: ok = False; break
            vc, pidx, optmoves, ml = lab
            fens.append(b.fen()); vcs.append(vc); pols.append(pidx); mls.append(ml)
            if step < k:
                if b.is_game_over(): break    # terminal: shorter sequence
                mv = rng.choice(optmoves) if rng.random() < 0.6 else rng.choice(list(b.legal_moves))
                acts.append(_move_to_action(mv, b.turn)); b.push(mv)
        if ok and len(fens) >= 2:
            out.append((fens, acts, vcs, pols, mls))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=str, required=True); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    per = math.ceil(a.n / a.workers); ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(SYZYGY,)) as pool:
        chunks = pool.map(_worker, [(per, a.k, 2000 + i) for i in range(a.workers)])
    data = [x for c in chunks for x in c][:a.n]
    pickle.dump(data, open(a.out, "wb"))
    avglen = sum(len(s[0]) for s in data) / len(data)
    print(f"saved {len(data)} sequences (k={a.k}, mean len {avglen:.2f}) to {a.out}")

if __name__ == "__main__":
    main()
