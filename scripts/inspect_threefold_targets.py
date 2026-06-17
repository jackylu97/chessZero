"""Inspect the actual value TARGETS the trainer would compute for a self-play game
that ended in threefold repetition, using the real make_target path.

Run: .venv/bin/python scripts/inspect_threefold_targets.py --buf <path.buf>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer


def wdl_scalar(w, d, l, draw_score):
    return w - l + draw_score * d


def targets_for_game(gh, c, rep_pen):
    """Per-ply WDL value target via the real make_target, with rep_pen overridable."""
    out = []
    n = len(gh.actions)
    for ply in range(n):
        _o, _m, _a, values, _r, _p = gh.make_target(
            ply, 0, c.td_steps, c.discount, ChessGame().action_space_size,
            value_head_type=c.value_head_type, history_frames=c.history_frames,
            eval_to_wdl_alpha=c.eval_to_wdl_alpha, eval_to_wdl_beta=c.eval_to_wdl_beta,
            q_ratio=c.q_ratio, warmstart_q_ratio=c.warmstart_q_ratio,
            selfplay_q_ratio=c.selfplay_q_ratio,
            repetition_penalty=rep_pen, repetition_penalty_window=c.repetition_penalty_window,
            repetition_penalty_decay=c.repetition_penalty_decay)
        v = np.asarray(values[0], dtype=float)  # (W, D, L)
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--max-games", type=int, default=1, help="how many threefold games to print")
    args = ap.parse_args()
    c = get_config("chess")
    game = ChessGame()
    rb = ReplayBuffer(c)
    rb.load(args.buf, game=game)
    games = rb.buffer
    sp_tf = [g for g in games
             if not g.external_values and g.draw_by_repetition and g.game_outcome == 0]
    print(f"buffer {args.buf}: {len(games)} games | "
          f"self-play threefold games: {len(sp_tf)}\n")
    if not sp_tf:
        print("no self-play threefold games found"); return

    for gi, gh in enumerate(sp_tf[-args.max_games:]):
        n = len(gh.actions)
        print(f"=== threefold game (len {n} plies, outcome {gh.game_outcome}, "
              f"draw_by_repetition={gh.draw_by_repetition}) ===")
        tgt_on = targets_for_game(gh, c, c.repetition_penalty)   # penalty ON (=0.35)
        tgt_off = targets_for_game(gh, c, 0.0)                   # penalty OFF (honest draw)
        print(f"{'ply':>4} {'root_v':>7} | {'W':>5} {'D':>5} {'L':>5} {'scalar':>7} (pen ON) "
              f"| {'W':>5} {'D':>5} {'L':>5} {'scalar':>7} (pen OFF)")
        for ply in range(n):
            rv = float(gh.root_values[ply]) if ply < len(gh.root_values) else float('nan')
            w, d, l = tgt_on[ply]
            w0, d0, l0 = tgt_off[ply]
            s_on = wdl_scalar(w, d, l, c.draw_score)
            s_off = wdl_scalar(w0, d0, l0, c.draw_score)
            # print every ply for short games, else sample head + tail
            if n <= 30 or ply < 4 or ply >= n - 12:
                print(f"{ply:>4} {rv:>+7.2f} | {w:>5.2f} {d:>5.2f} {l:>5.2f} {s_on:>+7.2f}        "
                      f"| {w0:>5.2f} {d0:>5.2f} {l0:>5.2f} {s_off:>+7.2f}")
            elif ply == 4:
                print(f"  ... (middle plies omitted) ...")
        # summary
        sc_on = np.array([wdl_scalar(*tgt_on[p], c.draw_score) for p in range(n)])
        sc_off = np.array([wdl_scalar(*tgt_off[p], c.draw_score) for p in range(n)])
        print(f"  mean value-target scalar:  pen ON = {sc_on.mean():+.3f}   "
              f"pen OFF = {sc_off.mean():+.3f}")
        print(f"  last-10-ply mean scalar:   pen ON = {sc_on[-10:].mean():+.3f}   "
              f"pen OFF = {sc_off[-10:].mean():+.3f}\n")


if __name__ == "__main__":
    main()
