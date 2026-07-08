"""Does MORE MCTS search (800 vs 200 sims) change self-play outcomes now that
the model is healthier? Earlier (weaker value head) more sims HURT — it amplified
a miscalibrated value. This re-runs the REAL GPU-resident self-play path with the
current checkpoint at several sim budgets and reports draw / threefold / decisive
/ conversion, holding everything but num_simulations fixed.

Outcomes are RAW (play_games_parallel_gpu_resident does not apply resignation),
so the draw/threefold rates are the model's true play, not adjudicated.

Run:
  .venv/bin/python scripts/probe_selfplay_sims.py \
      --checkpoint checkpoints/chess/<run>/checkpoint_40000.pt \
      --game chess_small --sims 200 800 --games 128 --device cuda
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
    m = 0
    for pt, val in PIECE_VAL.items():
        m += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return m


def classify(histories, game, max_plies):
    """Per game: outcome class + peak material edge (abs) + decisive bool +
    mean root_value at positions where the mover is up >= 5."""
    n = len(histories)
    draws = threefold = fifty = stale = insuff = plycap = decisive = 0
    peak_edges = []          # abs peak material edge per game
    decisive_flag = []       # did game end decisively
    rv_up5 = []              # root_value (mover POV) when mover up >= 5
    lengths = []
    for g in histories:
        actions = list(getattr(g, "actions", []) or [])
        rvs = list(getattr(g, "root_values", []) or [])
        outcome = float(getattr(g, "game_outcome", 0.0))
        lengths.append(len(actions))
        s = game.reset()
        boards = [s.board.copy()]
        ok = True
        for a in actions:
            try:
                s, _, _ = game.step(s, a)
            except Exception:
                ok = False; break
            boards.append(s.board.copy())
        if not ok:
            continue
        # peak material + value-at-winning
        peak = 0.0
        for t, b in enumerate(boards[:-1]):
            mw = material_white(b)
            mover_edge = mw if b.turn == chess.WHITE else -mw
            peak = max(peak, abs(mw))
            if mover_edge >= 5 and t < len(rvs):
                rv_up5.append(float(rvs[t]))
        peak_edges.append(peak)
        dec = outcome != 0.0
        decisive_flag.append(dec)
        if dec:
            decisive += 1
            continue
        draws += 1
        final = boards[-1]
        if final.is_repetition(3):
            threefold += 1
        elif final.halfmove_clock >= 100:
            fifty += 1
        elif final.is_stalemate():
            stale += 1
        elif final.is_insufficient_material():
            insuff += 1
        elif len(actions) >= max_plies:
            plycap += 1
    peak_edges = np.array(peak_edges)
    decisive_flag = np.array(decisive_flag)
    reach3 = peak_edges >= 3
    conv3 = decisive_flag[reach3].mean() if reach3.sum() else float("nan")
    reach5 = peak_edges >= 5
    conv5 = decisive_flag[reach5].mean() if reach5.sum() else float("nan")
    return {
        "n": n, "draw": draws, "threefold": threefold, "fifty": fifty,
        "stale": stale, "insuff": insuff, "plycap": plycap, "decisive": decisive,
        "avg_len": float(np.mean(lengths)) if lengths else 0.0,
        "reach3": int(reach3.sum()), "conv3": conv3,
        "reach5": int(reach5.sum()), "conv5": conv5,
        "rv_up5_mean": float(np.mean(rv_up5)) if rv_up5 else float("nan"),
        "rv_up5_n": len(rv_up5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--sims", type=int, nargs="+", default=[200, 800])
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--training-step", type=int, default=40000)
    args = ap.parse_args()

    dev = args.device
    game = ChessGame()
    cfg = get_config(args.game)
    cfg.device = dev
    # Match the live run: rep-penalty on. (resignation is applied downstream, not here.)
    cfg.root_terminal_draws = True
    cfg.root_terminal_draws_min_repeats = 2
    max_plies = int(getattr(cfg, "max_plies", 750))

    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    step = ckpt.get("step", "?")
    print(f"checkpoint step {step} ({os.path.basename(args.checkpoint)}), {args.games} games/setting, "
          f"rep-penalty ON, training_step={args.training_step}\n")

    rows = []
    for sims in args.sims:
        cfg.num_simulations = sims
        hist = play_games_parallel_gpu_resident(net, cfg, args.games, dev, training_step=args.training_step)
        r = classify(hist, game, max_plies)
        r["sims"] = sims
        rows.append(r)
        print(f"[sims={sims}] done")

    N = args.games
    print(f"\n{'sims':>5} {'draw%':>6} {'3fold%':>7} {'50mv%':>6} {'plycap%':>7} "
          f"{'decis%':>7} {'avglen':>7} {'reach+3':>8} {'conv+3':>7} {'reach+5':>8} "
          f"{'conv+5':>7} {'rv@+5':>7}")
    for r in rows:
        print(f"{r['sims']:>5} {r['draw']/N:>6.1%} {r['threefold']/N:>7.1%} "
              f"{r['fifty']/N:>6.1%} {r['plycap']/N:>7.1%} {r['decisive']/N:>7.1%} "
              f"{r['avg_len']:>7.0f} {r['reach3']:>8d} "
              f"{(r['conv3'] if not np.isnan(r['conv3']) else 0):>7.1%} "
              f"{r['reach5']:>8d} "
              f"{(r['conv5'] if not np.isnan(r['conv5']) else 0):>7.1%} "
              f"{(r['rv_up5_mean'] if not np.isnan(r['rv_up5_mean']) else 0):>+7.3f}")
    print("\n  reach+N = games where a side reached >= +N material at some ply;")
    print("  conv+N  = fraction of those that ended decisively (conversion rate);")
    print("  rv@+5   = mean stored root_value (mover POV) at positions where mover is up >= 5.")


if __name__ == "__main__":
    main()
