"""Value-target audit — falsifiable test of "V^pi ~= draw".

Hypothesis: the self-play value target is the game OUTCOME under the current
(weak) policy. If the policy can't convert wins, objectively-winning positions
are played out to draws, so their value target is "draw" (0). The value head
then correctly learns "winning position -> draw", losing all ranking signal.

Test: from the actual replay buffer, take SELF-PLAY positions (not warmstart),
reconstruct each board, ask Stockfish how winning it is for the side to move,
and bin by SF eval. For each bin report the value TARGET the position received:
the game outcome z (STM POV) and the stored MCTS root_value (STM POV).

If the ">+3 winning" bin shows mean z ~= 0 and a high draw fraction, V^pi~=draw
is confirmed: the buffer is teaching the value that winning positions draw.

Run: .venv/bin/python scripts/probe_value_target_audit.py --buf <path.buf>
"""
import argparse, os, sys, random, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import chess, chess.engine

from src.games.chess import _action_to_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--num-positions", type=int, default=400)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--min-ply", type=int, default=8, help="skip random-opening plies")
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # Read compact records directly (we only need actions/game_outcome/root_values;
    # skip the expensive from_compact_dict observation replay).
    recs = []
    with open(args.buf, "rb") as f:
        header = pickle.load(f)
        for _ in range(header["n_records"]):
            d, _prio = pickle.load(f)
            recs.append(d)
    sp = [d for d in recs if len(d.get("external_values", [])) == 0]
    warm = len(recs) - len(sp)
    draw_sp = sum(1 for d in sp if float(d["game_outcome"]) == 0.0)
    print(f"Loaded {args.buf}: {len(recs)} games ({len(sp)} self-play, {warm} warmstart)")
    print(f"Self-play draw rate: {draw_sp/max(1,len(sp)):.0%}\n")

    # Reconstruct boards by replaying actions from the start; collect candidates.
    cands = []  # (fen, z_stm, root_value)
    order = list(range(len(sp))); random.shuffle(order)
    for gi in order:
        d = sp[gi]
        outcome = float(d["game_outcome"]); rvs = d["root_values"]
        board = chess.Board()
        for i, a in enumerate(d["actions"]):
            if i >= args.min_ply:
                stm_white = board.turn == chess.WHITE
                z = outcome if stm_white else -outcome
                rv = float(rvs[i]) if i < len(rvs) else float("nan")
                cands.append((board.fen(), z, rv))
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
        if len(cands) > 6000:
            break
    if not cands:
        print("no replayable self-play positions found"); return
    sample = random.sample(cands, min(args.num_positions, len(cands)))
    print(f"Sampling {len(sample)} positions (of {len(cands)} candidates), SF depth {args.sf_depth}\n")

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    rows = []  # (sf_pawns, z, rv)
    for fen, z, rv in sample:
        b = chess.Board(fen)
        if b.is_game_over():
            continue
        info = eng.analyse(b, chess.engine.Limit(depth=args.sf_depth))
        cp = info["score"].pov(b.turn).score(mate_score=10000) / 100.0
        rows.append((cp, z, rv))
    eng.quit()

    rows = np.array(rows, dtype=float)
    cp, z, rv = rows[:, 0], rows[:, 1], rows[:, 2]
    bins = [(-1e9, -3, "losing < -3"), (-3, -1, "-3..-1"), (-1, 1, "equal -1..1"),
            (1, 3, "+1..+3"), (3, 1e9, "WINNING > +3")]
    print(f"{'SF eval bin (STM pawns)':>24} {'n':>5} {'mean z (target)':>16} "
          f"{'% drawn (z=0)':>14} {'mean root_value':>16}")
    for lo, hi, label in bins:
        m = (cp > lo) & (cp <= hi)
        if m.sum() == 0:
            continue
        zb = z[m]; rvb = rv[m][~np.isnan(rv[m])]
        print(f"{label:>24} {int(m.sum()):>5} {zb.mean():>+16.2f} "
              f"{np.mean(zb == 0):>13.0%} {(rvb.mean() if len(rvb) else float('nan')):>+16.3f}")
    print(f"\n  overall corr(SF eval, z target): {np.corrcoef(cp, z)[0,1]:+.2f}   "
          f"corr(SF eval, root_value): {np.corrcoef(cp, rv)[0,1] if not np.isnan(rv).any() else float('nan'):+.2f}")
    print("  Confirms V^pi~=draw if the WINNING>+3 bin has mean z ~0 and high % drawn.")


if __name__ == "__main__":
    main()
