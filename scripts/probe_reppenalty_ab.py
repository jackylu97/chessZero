"""Causal test: does the root-terminal-draws override actually reduce the
WINNING side's repetitions in self-play, or is it inert?

Runs the real GPU-resident self-play path twice with everything fixed except
config.root_terminal_draws (ON vs OFF), and compares threefold rate and the
number of times the side that is AHEAD in material plays a reversible move that
enters a repetition. If ON << OFF the override fires and helps; if ON ~= OFF the
override has nothing to steer toward (the conversion/value problem dominates).

Run:
  .venv/bin/python scripts/probe_reppenalty_ab.py \
      --checkpoint checkpoints/chess/<run>/checkpoint_40000.pt --games 128 --sims 200
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.self_play import play_games_parallel_gpu_resident
from scripts.eval_checkpoint_health import build_network

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}


def material_white(board):
    return sum(v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
               for pt, v in PIECE_VAL.items())


def analyze(histories, game):
    n = len(histories)
    threefold = decisive = draws = 0
    win_rep = lose_rep = 0          # winning/losing side reversible rep-entry events
    win_rep_games = 0              # games with >=1 winning-side rep entry
    for g in histories:
        actions = list(getattr(g, "actions", []) or [])
        outcome = float(getattr(g, "game_outcome", 0.0))
        s = game.reset(); boards = [s.board.copy()]; ok = True
        for a in actions:
            try:
                s, _, _ = game.step(s, a)
            except Exception:
                ok = False; break
            boards.append(s.board.copy())
        if not ok:
            continue
        if outcome != 0.0:
            decisive += 1
        else:
            draws += 1
            if boards[-1].is_repetition(3):
                threefold += 1
        had_win_rep = False
        for t in range(len(boards) - 1):
            before, after = boards[t], boards[t + 1]
            if not after.is_repetition(2):
                continue
            if material_white(after) != material_white(before):   # capture/promo => irreversible
                continue
            mw = material_white(before)
            edge = mw if before.turn == chess.WHITE else -mw
            if edge >= 1:
                win_rep += 1; had_win_rep = True
            elif edge <= -1:
                lose_rep += 1
        if had_win_rep:
            win_rep_games += 1
    return {"n": n, "threefold": threefold, "decisive": decisive, "draws": draws,
            "win_rep": win_rep, "lose_rep": lose_rep, "win_rep_games": win_rep_games}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--training-step", type=int, default=40000)
    args = ap.parse_args()

    dev = args.device
    game = ChessGame()
    cfg = get_config(args.game); cfg.device = dev
    cfg.num_simulations = args.sims
    cfg.root_terminal_draws_min_repeats = 2

    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    print(f"checkpoint step {ckpt.get('step','?')}, {args.games} games, sims {args.sims}\n")

    rows = {}
    for label, flag in [("OFF", False), ("ON", True)]:
        cfg.root_terminal_draws = flag
        hist = play_games_parallel_gpu_resident(net, cfg, args.games, dev, training_step=args.training_step)
        rows[label] = analyze(hist, game)
        print(f"[rep-penalty {label}] done")

    N = args.games
    print(f"\n{'rep-penalty':>12} {'3fold%':>7} {'decis%':>7} {'win-rep evts':>13} "
          f"{'games w/ win-rep':>17} {'lose-rep evts':>14}")
    for label in ("OFF", "ON"):
        r = rows[label]
        print(f"{label:>12} {r['threefold']/N:>7.1%} {r['decisive']/N:>7.1%} "
              f"{r['win_rep']:>13d} {r['win_rep_games']/N:>16.1%} {r['lose_rep']:>14d}")
    print("\n  win-rep evts = times the side AHEAD in material plays a reversible move that")
    print("  enters a repetition. The override is meant to drive this toward 0 for the")
    print("  winning side. If ON ~= OFF, the override has no better-valued move to steer to.")


if __name__ == "__main__":
    main()
