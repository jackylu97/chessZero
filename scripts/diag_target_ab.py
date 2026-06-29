"""A/B: same positions, compute the ACTUAL target_V WITH and WITHOUT the
repetition_penalty (and with/without selfplay_q). Quantifies how much the rep
penalty degrades corr(SF eval, target) on winning positions.
Caches SF evals across configs (single SF pass).
"""
import sys, random
sys.path.insert(0, "/workspace/chessZero")
import numpy as np
import chess, chess.engine

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer
from src.model.utils import WDL_W, WDL_L

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"


def target_V(g, ply, game, hf, **kw):
    _o, _m, _a, values, _r, _p = g.make_target(
        state_index=ply, num_unroll_steps=0, td_steps=-1, discount=1.0,
        action_space_size=game.action_space_size, value_head_type="wdl",
        history_frames=hf, **kw)
    wdl = np.asarray(values[0], dtype=np.float32)
    return float(wdl[WDL_W] - wdl[WDL_L])


def main():
    random.seed(1); np.random.seed(1)
    cfg = get_config("chess_small")
    game = ChessGame(); hf = cfg.history_frames
    rb = ReplayBuffer(max_size=200000); rb.load(BUF, game=game)
    games = rb.buffer

    cands = []
    order = list(range(len(games))); random.shuffle(order)
    for gi in order:
        g = games[gi]
        if getattr(g, "external_values", None): continue
        for p in range(8, len(g.actions)):
            cands.append((gi, p))
        if len(cands) > 20000: break
    sample = random.sample(cands, min(240, len(cands)))

    configs = {
        "pure-z (no pen, no q)": dict(repetition_penalty=0.0, selfplay_q_ratio=0.0),
        "q=0.1 only (no pen)":   dict(repetition_penalty=0.0, selfplay_q_ratio=0.1),
        "rep_pen=0.35 only":     dict(repetition_penalty=0.35, repetition_penalty_decay=0.93, selfplay_q_ratio=0.0),
        "CONFIG (pen+q, live)":  dict(repetition_penalty=0.35, repetition_penalty_decay=0.93, selfplay_q_ratio=0.1),
    }

    eng = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    rows = []
    for gi, ply in sample:
        g = games[gi]
        board = chess.Board(); ok = True
        for i in range(ply):
            mv = _action_to_move(int(g.actions[i]), board)
            if mv is None or mv not in board.legal_moves: ok = False; break
            board.push(mv)
        if not ok or board.is_game_over(): continue
        cp = eng.analyse(board, chess.engine.Limit(depth=10))["score"].pov(board.turn).score(mate_score=10000)/100.0
        tvs = {name: target_V(g, ply, game, hf, q_ratio=0.0, warmstart_q_ratio=None, **kw)
               for name, kw in configs.items()}
        rows.append((cp, tvs, 1.0 if getattr(g, "draw_by_repetition", False) else 0.0))
    eng.quit()

    cp = np.array([r[0] for r in rows])
    drep = np.array([r[2] for r in rows])
    print(f"n={len(rows)} positions   (rep-draw games: {drep.mean()*100:.0f}%)\n")
    print(f"{'config':<26} {'corr(SF,target)':>16} {'meanV WIN>+3':>13} {'%WIN taught LOSS':>18}")
    winm = cp > 3
    for name in configs:
        tv = np.array([r[1][name] for r in rows])
        corr = np.corrcoef(cp, tv)[0, 1]
        mw = tv[winm].mean() if winm.sum() else float("nan")
        taught_loss = np.mean(tv[winm] < -0.05) * 100 if winm.sum() else float("nan")
        print(f"{name:<26} {corr:>+16.3f} {mw:>+13.3f} {taught_loss:>17.0f}%")
    print(f"\nWIN>+3 count: {int(winm.sum())}")


if __name__ == "__main__":
    main()
