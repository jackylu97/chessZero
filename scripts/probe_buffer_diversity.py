"""Buffer diversity probe — measure self-play game/position diversity.

Reads compact .buf shards directly (no observation replay), measures:
  - first-N action-index opening-line diversity (unique lines, top-line share)
  - duplicate / near-duplicate whole games
  - played-move entropy per ply
  - game length distribution + outcome / draw-type breakdown
  - position-space coverage via FEN reconstruction (python-chess) on a sample

Run: .venv/bin/python scripts/probe_buffer_diversity.py <buf1> [buf2 ...]
"""
import argparse, os, sys, pickle, math, hashlib
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def load_records(path):
    recs = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        n = header["n_records"]
        for _ in range(n):
            d, _prio = pickle.load(f)
            recs.append(d)
    return header, recs


def entropy(counter):
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / tot
        h -= p * math.log2(p)
    return h


def summarize(path, fen_sample=0, sf_engine=None):
    header, recs = load_records(path)
    sp = [d for d in recs if len(d.get("external_values", [])) == 0]
    warm = len(recs) - len(sp)
    n = len(sp)
    print("=" * 92)
    print(f"{os.path.basename(path)}  total_games={len(recs)}  self_play={n}  warmstart={warm}")

    lengths = np.array([len(d["actions"]) for d in sp])
    outcomes = np.array([float(d["game_outcome"]) for d in sp])
    rep = np.array([bool(d.get("draw_by_repetition", False)) for d in sp])
    nopr = np.array([bool(d.get("draw_by_no_progress", False)) for d in sp])
    draws = outcomes == 0.0
    print(f"  game length: mean={lengths.mean():.1f} median={np.median(lengths):.0f} "
          f"min={lengths.min()} max={lengths.max()} std={lengths.std():.1f}")
    print(f"  outcomes: win(+1)={np.mean(outcomes>0):.1%} loss(-1)={np.mean(outcomes<0):.1%} "
          f"draw(0)={np.mean(draws):.1%}")
    print(f"  draw-type: by_repetition={rep.mean():.1%} by_no_progress={nopr.mean():.1%} "
          f"other_draw={np.mean(draws & ~rep & ~nopr):.1%}")

    # ---- opening-line diversity (first-N action indices) ----
    for N in (4, 8, 12, 20):
        lines = Counter()
        for d in sp:
            a = tuple(int(x) for x in d["actions"][:N])
            if len(a) == N:
                lines[a] += 1
        cnt = sum(lines.values())
        if cnt == 0:
            continue
        uniq = len(lines)
        top = lines.most_common(1)[0][1]
        # how many games to cover 50%/90% of mass
        svals = sorted(lines.values(), reverse=True)
        cum = np.cumsum(svals)
        c50 = int(np.searchsorted(cum, 0.5 * cnt) + 1)
        c90 = int(np.searchsorted(cum, 0.9 * cnt) + 1)
        Hmax = math.log2(uniq) if uniq > 1 else 1.0
        print(f"  first-{N:>2} plies: unique_lines={uniq:>4}/{cnt}  "
              f"top_line_share={top/cnt:.1%}  lines_for_50%={c50}  lines_for_90%={c90}  "
              f"line_entropy={entropy(lines):.2f}b (norm {entropy(lines)/Hmax if Hmax>0 else 0:.2f})")

    # ---- whole-game duplicate fraction (full action seq hash) ----
    full_hashes = Counter()
    for d in sp:
        h = hashlib.md5(np.asarray(d["actions"], dtype=np.int32).tobytes()).hexdigest()
        full_hashes[h] += 1
    dup_games = sum(c for c in full_hashes.values() if c > 1) - sum(1 for c in full_hashes.values() if c > 1)
    print(f"  EXACT duplicate games: {len(sp)-len(full_hashes)} dup instances "
          f"({(len(sp)-len(full_hashes))/max(1,len(sp)):.1%}); unique full games={len(full_hashes)}")

    # near-duplicate: share first 20 plies AND same outcome AND length within 4
    nd_buckets = defaultdict(list)
    for d in sp:
        key = (tuple(int(x) for x in d["actions"][:20]), float(d["game_outcome"]))
        nd_buckets[key].append(len(d["actions"]))
    near_dup = 0
    for key, lens in nd_buckets.items():
        if len(lens) > 1:
            near_dup += len(lens) - 1
    print(f"  near-dup games (same first-20 plies + outcome): {near_dup} "
          f"({near_dup/max(1,len(sp)):.1%})")

    # ---- played-move entropy per ply (positions 0..29) ----
    perply = defaultdict(Counter)
    for d in sp:
        for ply, a in enumerate(d["actions"][:30]):
            perply[ply][int(a)] += 1
    ent_per_ply = [(ply, entropy(perply[ply]), len(perply[ply]), sum(perply[ply].values()))
                   for ply in sorted(perply)]
    mean_ent = np.mean([e for _, e, _, _ in ent_per_ply])
    print(f"  played-move entropy per ply (first 30): mean={mean_ent:.2f} bits")
    for ply in (0, 1, 2, 4, 8, 16, 29):
        if ply < len(ent_per_ply):
            _, e, k, c = ent_per_ply[ply]
            print(f"      ply {ply:>2}: entropy={e:.2f}b  unique_moves={k}  (n={c})")

    # ---- unique positions vs total (FEN reconstruction on sample) ----
    if fen_sample > 0:
        import chess
        from src.games.chess import _action_to_move
        import random
        random.seed(0)
        idxs = list(range(n))
        random.shuffle(idxs)
        idxs = idxs[:fen_sample]
        seen_fen = set()
        seen_fen_nomovenum = set()
        total_pos = 0
        sf_winning_draw = 0
        sf_checked = 0
        for gi in idxs:
            d = sp[gi]
            board = chess.Board()
            for a in d["actions"]:
                # FEN without halfmove/fullmove counters (position identity)
                parts = board.fen().split(" ")
                seen_fen.add(board.fen())
                seen_fen_nomovenum.add(" ".join(parts[:4]))
                total_pos += 1
                mv = _action_to_move(int(a), board)
                if mv is None or mv not in board.legal_moves:
                    break
                board.push(mv)
        print(f"  position coverage ({len(idxs)} games): total_positions={total_pos}  "
              f"unique_FEN={len(seen_fen)} ({len(seen_fen)/max(1,total_pos):.1%})  "
              f"unique_position(no-movenum)={len(seen_fen_nomovenum)} "
              f"({len(seen_fen_nomovenum)/max(1,total_pos):.1%})")

    return dict(n=n, mean_len=float(lengths.mean()), draw_rate=float(np.mean(draws)),
                rep_rate=float(rep.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bufs", nargs="+")
    ap.add_argument("--fen-sample", type=int, default=0,
                    help="reconstruct FENs for N sampled games (position coverage)")
    args = ap.parse_args()
    for p in args.bufs:
        summarize(p, fen_sample=args.fen_sample)
        print()


if __name__ == "__main__":
    main()
