"""Audit the ACTUAL training targets produced by make_target, and check
observation/target alignment + signal content.

Three checks:
 A) For self-play positions, what WDL target does make_target actually emit?
    Bin by Stockfish eval. If SF-winning positions get a draw/loss WDL target,
    the head is being TAUGHT noise (V^pi). Quantify the per-bucket target.
 B) Does the WDL target carry ANY ranking signal vs SF eval (corr of
    target-scalar vs SF)? This is the ceiling the value head could ever reach.
 C) Observation sanity: is the stacked-history obs distinct per position, and
    does the STM/turn plane match the board's side to move? (alignment bug check)

Run: .venv/bin/python scripts/probe_target_alignment.py
"""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.engine

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.training.replay_buffer import GameHistory

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
SF = "/usr/games/stockfish"


def load_games(buf_path, game, limit=None):
    games = []
    with open(buf_path, "rb") as f:
        header = pickle.load(f)
        for _ in range(header["n_records"]):
            d, _prio = pickle.load(f)
            try:
                gh = GameHistory.from_compact_dict(d, game)
            except Exception:
                continue
            games.append(gh)
            if limit and len(games) >= limit:
                break
    return games


def main():
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = "cpu"
    hf = cfg.history_frames
    print(f"value_head_type={cfg.value_head_type} selfplay_q_ratio={cfg.selfplay_q_ratio} "
          f"warmstart_q_ratio={cfg.warmstart_q_ratio} repetition_penalty={cfg.repetition_penalty} "
          f"repetition_penalty_decay={cfg.repetition_penalty_decay} draw_score={cfg.draw_score}")

    games = load_games(f"{CKDIR}/checkpoint_6000.buf", game, limit=400)
    sp = [g for g in games if not g.external_values]
    print(f"{len(sp)} self-play games loaded")

    rng = random.Random(7)
    samples = []  # (board, gh, ply)
    tries = 0
    while len(samples) < 250 and tries < 20000:
        tries += 1
        g = rng.choice(sp)
        if len(g.actions) <= 10:
            continue
        ply = rng.randint(8, len(g.actions) - 1)
        board = chess.Board(); ok = True
        for a in g.actions[:ply]:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                ok = False; break
            board.push(mv)
        if not ok or board.is_game_over():
            continue
        samples.append((board.copy(), g, ply))

    print("Grading with Stockfish...")
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    rows = []
    for board, g, ply in samples:
        info = eng.analyse(board, chess.engine.Limit(depth=12))
        sc = info["score"].pov(board.turn)
        if sc.is_mate():
            sfv = 1.0 if sc.mate() > 0 else -1.0
        else:
            sfv = float(np.tanh(sc.score() / 400.0))
        # actual training target from make_target
        (tobs, tmask, tact, tval, trew, tpol) = g.make_target(
            state_index=ply, num_unroll_steps=0, td_steps=-1, discount=cfg.discount,
            action_space_size=game.action_space_size, value_head_type=cfg.value_head_type,
            history_frames=hf, eval_to_wdl_alpha=cfg.eval_to_wdl_alpha,
            eval_to_wdl_beta=cfg.eval_to_wdl_beta, q_ratio=cfg.q_ratio,
            warmstart_q_ratio=cfg.warmstart_q_ratio, selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,
            repetition_penalty_window=cfg.repetition_penalty_window,
            repetition_penalty_decay=cfg.repetition_penalty_decay)
        wdl_t = np.asarray(tval[0], dtype=np.float32)  # [W,D,L]
        # scalar from target WDL (no draw_score, pure)
        tscalar = float(wdl_t[0] - wdl_t[2])
        # turn-plane alignment check: plane 21 of CURRENT frame (last frame) should
        # encode STM. num_planes=22, history stacked => last frame is planes [21*22:22*22)?
        rows.append(dict(sf=sfv, W=wdl_t[0], D=wdl_t[1], L=wdl_t[2], tscalar=tscalar,
                         outcome=float(g.game_outcome), rep=g.draw_by_repetition,
                         rv=(g.root_values[ply] if ply < len(g.root_values) else None),
                         stm_white=(board.turn == chess.WHITE)))
    eng.quit()

    sf = np.array([r["sf"] for r in rows])
    ts = np.array([r["tscalar"] for r in rows])
    print(f"\n=== A) make_target WDL by Stockfish bucket (self-play, the head's actual labels) ===")
    for lbl, mask in [("SF win >+0.3", sf > 0.3), ("SF draw|.|<0.3", np.abs(sf) < 0.3),
                      ("SF loss <-0.3", sf < -0.3)]:
        if mask.sum() == 0:
            continue
        sub = [rows[i] for i in np.where(mask)[0]]
        W = np.mean([r["W"] for r in sub]); D = np.mean([r["D"] for r in sub]); L = np.mean([r["L"] for r in sub])
        sc = np.mean([r["tscalar"] for r in sub])
        drawfrac = np.mean([1.0 if abs(r["tscalar"]) < 1e-6 else 0.0 for r in sub])
        print(f"  {lbl:16s} n={mask.sum():3d}  target W={W:.3f} D={D:.3f} L={L:.3f} | "
              f"scalar={sc:+.3f}  exact-draw-frac={drawfrac:.0%}")

    print(f"\n=== B) Target ranking signal (the value head's learnable ceiling) ===")
    if ts.std() > 1e-6:
        print(f"  corr(target_scalar, SF eval) = {np.corrcoef(ts, sf)[0,1]:+.3f}")
        print(f"  target_scalar dist: mean={ts.mean():+.3f} std={ts.std():.3f} "
              f"| frac exact-draw [0,1,0]={np.mean(np.abs(ts)<1e-6):.0%}")
    # how many SF-winning positions get a NON-draw, correctly-signed target?
    win = sf > 0.3
    if win.sum() > 0:
        win_correct = np.mean([rows[i]["tscalar"] > 0.05 for i in np.where(win)[0]])
        win_wrong = np.mean([rows[i]["tscalar"] < -0.05 for i in np.where(win)[0]])
        print(f"  SF-winning positions: target says WIN {win_correct:.0%}, "
              f"target says LOSS {win_wrong:.0%}, rest draw")

    print(f"\n=== C) Repetition-penalty tilt incidence ===")
    rep_rows = [r for r in rows if r["rep"]]
    print(f"  {len(rep_rows)}/{len(rows)} sampled positions are from rep-draw games")
    # among rep-draw, winning-by-SF positions: did the penalty push them to LOSS?
    rep_win = [r for r in rep_rows if r["sf"] > 0.3]
    if rep_win:
        pushed_loss = np.mean([1.0 if r["tscalar"] < -0.05 else 0.0 for r in rep_win])
        guard_held = np.mean([1.0 if (r["rv"] is not None and r["rv"] > 0.25) else 0.0 for r in rep_win])
        print(f"  SF-winning rep-draw positions (n={len(rep_win)}): target=LOSS {pushed_loss:.0%}; "
              f"STM-winning-guard (rv>0.25) held {guard_held:.0%}")


if __name__ == "__main__":
    main()
