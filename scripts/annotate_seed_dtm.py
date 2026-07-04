"""Annotate an endgame-seed FEN archive with Gaviota |DTM| (plies-to-mate).

Produces ``<archive>.dtm`` — one line per input FEN: ``<dtm_plies>`` (int),
``0`` for drawn seeds, ``-1`` where the probe fails. Consumed by the
DTM-stratified seed curriculum (config.seed_curriculum): early training samples
short-chain seeds (small DTM — the p^N chain math is benign there), annealing
toward the full archive as conversion mastery arrives (reverse curriculum,
Florensa et al. 2017; also how human endgame pedagogy orders material).

Run once (CPU, pooled):
  PYTHONPATH=. .venv/bin/python scripts/annotate_seed_dtm.py \
      --seeds data/endgame_seeds_train.txt --workers 8
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.gaviota

_TB = None


def _init(path: str):
    global _TB
    _TB = chess.gaviota.open_tablebase(path)


def _dtm(fen: str) -> int:
    try:
        board = chess.Board(fen)
        dtm = _TB.probe_dtm(board)
        return abs(int(dtm))  # 0 = draw
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/endgame_seeds_train.txt")
    ap.add_argument("--gaviota", default="data/gaviota")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    fens = [ln.strip() for ln in open(args.seeds) if ln.strip()]
    out_path = Path(args.seeds).with_suffix(Path(args.seeds).suffix + ".dtm")
    with Pool(args.workers, initializer=_init, initargs=(args.gaviota,)) as pool:
        dtms = pool.map(_dtm, fens, chunksize=256)
    with open(out_path, "w") as f:
        f.write("\n".join(str(d) for d in dtms) + "\n")

    import collections
    n_draw = sum(1 for d in dtms if d == 0)
    n_err = sum(1 for d in dtms if d < 0)
    won = [d for d in dtms if d > 0]
    hist = collections.Counter(min(d // 10, 9) for d in won)
    print(f"annotated {len(fens)} seeds -> {out_path}")
    print(f"  won: {len(won)}  drawn: {n_draw}  probe-fail: {n_err}")
    print("  DTM deciles (10-ply bins):",
          {f"{k*10}-{k*10+9}": hist[k] for k in sorted(hist)})


if __name__ == "__main__":
    main()
