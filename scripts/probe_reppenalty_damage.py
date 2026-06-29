"""Isolate the repetition_penalty's effect on the value target's ranking signal.

Compare make_target WDL with rep_penalty ON (config: 0.35, decay 0.93) vs OFF
(0.0), on the SAME self-play positions graded by Stockfish. Question: does the
penalty HELP (push lost-but-shuffled toward loss = correct) or HURT (push
won-but-shuffled toward loss = wrong)?

Metrics per setting:
 - corr(target_scalar, SF eval)
 - among SF-winning positions: frac labeled LOSS (wrong) / WIN (right)
 - among SF-losing  positions: frac labeled LOSS (right)
 - sign-agreement of target vs SF on decisive (|SF|>0.3) positions

Run: .venv/bin/python scripts/probe_reppenalty_damage.py
"""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import chess, chess.engine

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
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
                games.append(GameHistory.from_compact_dict(d, game))
            except Exception:
                continue
            if limit and len(games) >= limit:
                break
    return games


def target_scalar(g, ply, cfg, game, rep):
    out = g.make_target(state_index=ply, num_unroll_steps=0, td_steps=-1,
        discount=cfg.discount, action_space_size=game.action_space_size,
        value_head_type="wdl", history_frames=cfg.history_frames,
        eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
        q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
        selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=rep,
        repetition_penalty_window=cfg.repetition_penalty_window,
        repetition_penalty_decay=(cfg.repetition_penalty_decay if rep > 0 else 0.0))
    w = np.asarray(out[3][0], dtype=np.float32)
    return float(w[0] - w[2])


def main():
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = "cpu"
    games = load_games(f"{CKDIR}/checkpoint_6000.buf", game, limit=400)
    sp = [g for g in games if not g.external_values]
    rng = random.Random(11)
    samples = []
    while len(samples) < 250 and len(samples) < 100000:
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
        if len(samples) >= 250:
            break

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    sfs, on, off = [], [], []
    for board, g, ply in samples:
        info = eng.analyse(board, chess.engine.Limit(depth=12))
        sc = info["score"].pov(board.turn)
        sfv = (1.0 if sc.mate() > 0 else -1.0) if sc.is_mate() else float(np.tanh(sc.score()/400.0))
        sfs.append(sfv)
        on.append(target_scalar(g, ply, cfg, game, rep=0.35))
        off.append(target_scalar(g, ply, cfg, game, rep=0.0))
    eng.quit()
    sfs = np.array(sfs); on = np.array(on); off = np.array(off)

    for name, t in [("rep_penalty ON (0.35)", on), ("rep_penalty OFF (0.0)", off)]:
        corr = np.corrcoef(t, sfs)[0,1] if t.std() > 1e-6 else float("nan")
        win = sfs > 0.3; loss = sfs < -0.3
        win_as_loss = np.mean(t[win] < -0.05) if win.sum() else float("nan")
        win_as_win = np.mean(t[win] > 0.05) if win.sum() else float("nan")
        loss_as_loss = np.mean(t[loss] < -0.05) if loss.sum() else float("nan")
        dec = np.abs(sfs) > 0.3
        sign_agree = np.mean(np.sign(t[dec]) == np.sign(sfs[dec])) if dec.sum() else float("nan")
        print(f"\n{name}")
        print(f"  corr(target,SF)={corr:+.3f}  scalar mean={t.mean():+.3f} std={t.std():.3f}")
        print(f"  SF-WIN positions (n={win.sum()}): labeled WIN {win_as_win:.0%}  labeled LOSS {win_as_loss:.0%}")
        print(f"  SF-LOSS positions (n={loss.sum()}): labeled LOSS {loss_as_loss:.0%}")
        print(f"  decisive sign-agreement target vs SF: {sign_agree:.0%}")


if __name__ == "__main__":
    main()
