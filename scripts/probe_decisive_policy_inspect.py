"""Print the actual policy targets (moves + visit fractions) through the conversion
phase of LONG decisive self-play games — to see what the targets look like where the
model is clearly winning but grinding it out over many plies.

Run: .venv/bin/python scripts/probe_decisive_policy_inspect.py --buf <path.buf>
"""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import chess
from src.games.chess import _action_to_move


def decode_step_policy(item, board, topk=5):
    """item = (nz_indices, values) for one ply. Return [(uci, frac)] top-k by frac."""
    nz, vals = item
    pairs = []
    for a, v in zip(np.asarray(nz).tolist(), np.asarray(vals).tolist()):
        mv = _action_to_move(int(a), board)
        pairs.append((mv.uci() if mv is not None else f"?{a}", float(v)))
    pairs.sort(key=lambda x: -x[1])
    return pairs[:topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--tail", type=int, default=18)
    args = ap.parse_args()
    recs = []
    with open(args.buf, "rb") as f:
        header = pickle.load(f)
        for _ in range(header["n_records"]):
            d, _ = pickle.load(f)
            recs.append(d)
    sp = [d for d in recs if len(d.get("external_values", [])) == 0]
    dec = [d for d in sp if float(d["game_outcome"]) != 0.0 and d["policies_mode"] == "sparse"]
    dec = [d for d in dec if len(d["actions"]) >= args.min_len]
    dec.sort(key=lambda d: -len(d["actions"]))
    print(f"{len(dec)} decisive sparse games with len>={args.min_len}\n")

    for d in dec[:args.games]:
        actions = d["actions"]; rv = d["root_values"]; z = float(d["game_outcome"])
        pdata = d["policies_data"]; n = len(actions)
        # replay to get boards
        board = chess.Board(); boards = []
        ok = True
        for a in actions:
            boards.append(board.copy())
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                ok = False; break
            board.push(mv)
        print(f"==== decisive game len {n}, outcome {z:+.0f}, replay_ok={ok} ====")
        print(f"  final fen: {board.fen()}")
        start = max(8, n - args.tail)
        for ply in range(start, min(n, len(pdata), len(boards))):
            b = boards[ply]
            stm_white = (ply % 2 == 0)
            vw = (rv[ply] if (stm_white == (z > 0)) else -rv[ply]) if ply < len(rv) else float('nan')
            top = decode_step_policy(pdata[ply], b, topk=5)
            played = _action_to_move(int(actions[ply]), b)
            played_uci = played.uci() if played else "?"
            top_str = "  ".join(f"{u}:{f:.2f}" for u, f in top)
            print(f"  ply{ply:>3} {'w' if stm_white else 'b'} Vwin={vw:+.2f} played={played_uci:<6} | {top_str}")
        print()


if __name__ == "__main__":
    main()
