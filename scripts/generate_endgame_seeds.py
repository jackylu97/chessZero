"""Generate a diverse endgame-seed archive for on-policy seeding (Phase 1).

Random legal 3/4/5-man positions where the side to move is winning (Syzygy wdl=2),
binned by piece_count x white-material-signature so the archive spans diverse endgame
TYPES (KQvK, KRvK, KQvKR, KRPvKP, ...). Random placement within a material gives a
natural spread of distance-to-mate, and the model plays each seed OUT (covering the
DTM range itself), so the archive doesn't need the (slow) Gaviota DTM probe — the DTM
TARGET is computed by the prober during self-play. Keeps ~12% true tablebase DRAWS so
the model learns the won/drawn boundary. STM mirrored 50/50. Output is a flat,
bin-balanced FEN list (uniform sampling ~= bin-uniform) + a metadata json.

Run: PYTHONPATH=. .venv/bin/python scripts/generate_endgame_seeds.py \
        --out data/endgame_seeds --per-bin 600
"""
import argparse, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess, chess.syzygy

SYM = {'Q': chess.QUEEN, 'R': chess.ROOK, 'B': chess.BISHOP, 'N': chess.KNIGHT, 'P': chess.PAWN}
VAL = {'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1}
PC_WEIGHTS = [0.15, 0.40, 0.45]   # 3 / 4 / 5 man


def random_winning_board(rng):
    """Random legal position, white = stronger side, white to move. Returns
    (board, white_material_signature) or None."""
    pc = rng.choices([3, 4, 5], weights=PC_WEIGHTS)[0]
    extras = pc - 2
    for _ in range(12):
        wn = rng.randint(max(1, extras - 1), extras)
        bn = extras - wn
        we = [rng.choice("QRBNP") for _ in range(wn)]
        be = [rng.choice("QRBNP") for _ in range(bn)]
        if sum(VAL[p] for p in we) <= sum(VAL[p] for p in be):
            continue
        sqs = rng.sample(range(64), 2 + wn + bn)
        if chess.square_distance(sqs[0], sqs[1]) <= 1:
            continue
        b = chess.Board.empty()
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        ok = True; i = 2
        for col, ex in ((chess.WHITE, we), (chess.BLACK, be)):
            for p in ex:
                sq = sqs[i]; i += 1
                if p == 'P' and chess.square_rank(sq) in (0, 7):
                    ok = False; break
                b.set_piece_at(sq, chess.Piece(SYM[p], col))
            if not ok:
                break
        if not ok:
            continue
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_game_over():
            sig = "".join(sorted(we, key=lambda c: -VAL[c])) + "v" + \
                  "".join(sorted(be, key=lambda c: -VAL[c]))
            return b, sig
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/endgame_seeds")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--per-bin", type=int, default=600, help="target won FENs per (pc,material) bin")
    ap.add_argument("--draw-frac", type=float, default=0.12)
    ap.add_argument("--max-attempts", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    syz = chess.syzygy.open_tablebase(args.syzygy)

    won: dict = {}      # (pc, sig) -> [fen]
    draws: dict = {3: [], 4: [], 5: []}
    n_won = lambda: sum(len(v) for v in won.values())
    draw_cap = lambda: int(args.draw_frac / (1 - args.draw_frac) * n_won() / 3) + 1
    attempts = 0
    while attempts < args.max_attempts:
        attempts += 1
        if attempts % 200000 == 0:
            print(f"  ...{attempts} attempts | won={n_won()} ({len(won)} bins) | "
                  f"draws={sum(len(v) for v in draws.values())}", flush=True)
        r = random_winning_board(rng)
        if r is None:
            continue
        b, sig = r
        pc = len(b.piece_map())
        try:
            wdl = syz.probe_wdl(b)
        except Exception:
            continue
        fen = b.mirror().fen() if rng.random() < 0.5 else b.fen()  # STM-mirror 50/50
        if wdl == 2:
            key = (pc, sig)
            lst = won.setdefault(key, [])
            if len(lst) < args.per_bin:
                lst.append(fen)
        elif wdl == 0 and len(draws[pc]) < draw_cap():
            draws[pc].append(fen)
    syz.close()

    fens, meta = [], []
    for (pc, sig), lst in won.items():
        for f in lst:
            fens.append(f); meta.append({"fen": f, "pc": pc, "mat": sig, "wdl": 2})
    for pc, lst in draws.items():
        for f in lst:
            fens.append(f); meta.append({"fen": f, "pc": pc, "mat": "draw", "wdl": 0})
    rng.shuffle(fens)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".txt", "w") as fp:
        fp.write("\n".join(fens) + "\n")
    with open(args.out + ".meta.json", "w") as fp:
        json.dump(meta, fp)
    # report
    by_pc = {3: 0, 4: 0, 5: 0}
    for (pc, _), lst in won.items():
        by_pc[pc] += len(lst)
    print(f"\nattempts={attempts} | total seeds={len(fens)} | won bins={len(won)}")
    print(f"won by pc: {by_pc} | draws: {{k: len(v) for k,v in draws.items()}}".replace("{k: len(v) for k,v in draws.items()}", str({k: len(v) for k, v in draws.items()})))
    print(f"sample material bins: {sorted(set(s for _, s in won))[:12]}")
    print(f"saved -> {args.out}.txt (+ .meta.json)")


if __name__ == "__main__":
    main()
