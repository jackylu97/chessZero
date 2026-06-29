"""Why does the WINNING side allow threefold repetitions despite the
root-terminal-draws penalty (min_repeats=2)?

The MCTS override (tensor_mcts._select, ~line 1130) pins any root move that
completes a repetition to the normalized draw_score. The code comment states the
design assumption: "a winning side's other children normalize ABOVE draw_score
(so it avoids the repeat)". That only deters the winning side if the value head
rates SOME non-repeating move above a draw. This probe tests that on the actual
buffer games: at the ply where the winning side enters a repetition, what did the
stored MCTS root_value say, bucketed by how much material that side was up?

If root_value ~= 0 (draw) while the side is +3/+5/+9 in material, the override is
inert: the repeating move (pinned to 0) ties every alternative, so there is no
deterrent and the side shuffles. No net re-evaluation needed — root_values are
stored per ply in GameHistory.

Run:
  .venv/bin/python scripts/probe_threefold_indifference.py \
      --buffer checkpoints/chess/<run>/checkpoint_41000.buf --max-games 4000
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch  # noqa: F401  (needed to unpickle GameHistory observation tensors)
import chess

from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}


def material_white(board):
    m = 0
    for pt, val in PIECE_VAL.items():
        m += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--max-games", type=int, default=4000)
    args = ap.parse_args()

    game = ChessGame()
    print(f"loading {args.buffer} ...")
    rb = ReplayBuffer(max_size=10_000_000)
    rb.load(args.buffer, game=game)
    games = rb.buffer[: args.max_games]
    print(f"  {len(games)} games loaded\n")

    n_games = 0
    n_threefold = 0
    n_other_draw = 0
    n_decisive = 0
    # rep-entry events by the side that is materially ahead (winning side):
    win_entries = []   # (material_edge_of_mover, root_value_of_mover, reversible_repeat_bool)
    # also track when the LOSING side enters a repetition (expected/correct)
    lose_entries = []

    for g in games:
        n_games += 1
        actions = list(getattr(g, "actions", []) or [])
        rvs = list(getattr(g, "root_values", []) or [])
        outcome = float(getattr(g, "game_outcome", 0.0))
        if not actions:
            continue

        # Replay through python-chess, tracking material + per-ply boards.
        s = game.reset()
        boards = [s.board.copy()]
        ok = True
        for a in actions:
            try:
                s, _, _ = game.step(s, a)
            except Exception:
                ok = False
                break
            boards.append(s.board.copy())
        if not ok or len(boards) < 4:
            continue

        final = boards[-1]
        is_threefold = (outcome == 0.0) and final.is_repetition(3)
        if outcome != 0.0:
            n_decisive += 1
            continue
        if is_threefold:
            n_threefold += 1
        else:
            n_other_draw += 1
            continue

        # For a threefold game, scan plies; detect the move that *enters* a
        # repetition (resulting position now occurs >= 2 times) and is reversible.
        # boards[t] = position before move t (side to move = boards[t].turn).
        for t in range(len(boards) - 1):
            before = boards[t]
            after = boards[t + 1]
            # resulting position repeats a prior one?
            if not after.is_repetition(2):
                continue
            mover_is_white = before.turn == chess.WHITE
            mat_w = material_white(before)
            mover_edge = mat_w if mover_is_white else -mat_w
            # reversible repeat? (no material change, mover piece not a pawn)
            move = before.peek() if False else None
            reversible = (material_white(after) == mat_w)  # no capture/promotion
            # mover root value (mover POV) at this ply
            rv = float(rvs[t]) if t < len(rvs) else float("nan")
            if np.isnan(rv):
                continue
            if mover_edge >= 1:        # mover is the side that is ahead
                win_entries.append((mover_edge, rv, reversible))
            elif mover_edge <= -1:     # mover is the side that is behind
                lose_entries.append((-mover_edge, rv, reversible))

    print(f"buffer: {args.buffer}")
    print(f"games scanned: {n_games}")
    print(f"  threefold draws: {n_threefold}   other draws: {n_other_draw}   decisive: {n_decisive}")
    print(f"  threefold rate (of all): {n_threefold/max(1,n_games):.1%}\n")

    def summarize(label, entries):
        if not entries:
            print(f"  {label}: (no events)")
            return
        edges = np.array([e for e, _, _ in entries])
        rvs = np.array([r for _, r, _ in entries])
        revs = np.array([1 if rev else 0 for _, _, rev in entries])
        print(f"  {label}: {len(entries)} rep-entry events")
        print(f"     reversible (mask-eligible) repeats: {revs.mean():.0%}")
        for lo, hi in [(1, 2), (3, 4), (5, 8), (9, 99)]:
            m = (edges >= lo) & (edges <= hi)
            if m.sum() == 0:
                continue
            print(f"     mover up {lo}-{hi if hi<99 else '+'} pawns: n={m.sum():4d}  "
                  f"mean root_value={rvs[m].mean():+.3f}  "
                  f"(|rv|<0.2: {np.mean(np.abs(rvs[m])<0.2):.0%})")

    print("WINNING side enters a repetition (mover is the side AHEAD in material):")
    print("  >>> if root_value ~= 0 here, the rep-penalty override has nothing to bite on")
    summarize("winning-side rep entries", win_entries)
    print("\nLOSING side enters a repetition (expected/correct — banking the draw):")
    summarize("losing-side rep entries", lose_entries)


if __name__ == "__main__":
    main()
