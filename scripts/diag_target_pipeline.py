"""Definitive value-target audit: compute the ACTUAL training target
(make_target WDL with repetition_penalty + selfplay_q_ratio applied), decode it
to a scalar, and bin by Stockfish eval. Answers: does the rep penalty deliver
CORRECT decisive signal, or does it TILT WINNING positions toward loss (a bug)?
"""
import argparse, sys, random, pickle
sys.path.insert(0, "/workspace/chessZero")
import numpy as np
import chess, chess.engine

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer
from src.model.utils import WDL_W, WDL_D, WDL_L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", default="/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")
    ap.add_argument("--num-positions", type=int, default=240)
    ap.add_argument("--sf-depth", type=int, default=10)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    cfg = get_config("chess_small")
    game = ChessGame()
    rb = ReplayBuffer(max_size=200000)
    rb.load(args.buf, game=game)
    games = rb.buffer
    print(f"loaded {len(games)} games; rep_pen={cfg.repetition_penalty} decay={cfg.repetition_penalty_decay} "
          f"selfplay_q={cfg.selfplay_q_ratio} draw_score(eval-only, not in target)={cfg.draw_score}")

    # collect candidate (game_idx, ply) for self-play (no external_values) positions
    cands = []
    order = list(range(len(games))); random.shuffle(order)
    for gi in order:
        g = games[gi]
        if getattr(g, "external_values", None):
            continue
        n_actions = len(g.actions)
        for p in range(args.min_ply, n_actions):
            cands.append((gi, p))
        if len(cands) > 20000:
            break
    sample = random.sample(cands, min(args.num_positions, len(cands)))
    print(f"sampling {len(sample)} self-play positions, SF depth {args.sf_depth}\n")

    def target_scalar_wdl(g, ply):
        """Compute make_target's WDL target at ply, decode to scalar (W - L),
        replicating cfg's rep penalty + selfplay_q blend."""
        obs_list, obs_mask, actions, values, rewards, policies = g.make_target(
            state_index=ply, num_unroll_steps=0, td_steps=cfg.td_steps,
            discount=cfg.discount, action_space_size=game.action_space_size,
            value_head_type="wdl", history_frames=cfg.history_frames,
            eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
            q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
            selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,
            repetition_penalty_window=cfg.repetition_penalty_window,
            repetition_penalty_decay=cfg.repetition_penalty_decay,
        )
        wdl = np.asarray(values[0], dtype=np.float32)  # (3,) W,D,L
        scalar = float(wdl[WDL_W] - wdl[WDL_L])
        return scalar, wdl

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    rows = []  # (cp, target_scalar, pW, pD, pL, draw_by_rep)
    for gi, ply in sample:
        g = games[gi]
        board = chess.Board()
        ok = True
        for i in range(ply):
            mv = _action_to_move(int(g.actions[i]), board)
            if mv is None or mv not in board.legal_moves:
                ok = False; break
            board.push(mv)
        if not ok or board.is_game_over():
            continue
        info = eng.analyse(board, chess.engine.Limit(depth=args.sf_depth))
        cp = info["score"].pov(board.turn).score(mate_score=10000) / 100.0
        ts, wdl = target_scalar_wdl(g, ply)
        rows.append((cp, ts, float(wdl[0]), float(wdl[1]), float(wdl[2]),
                     1.0 if getattr(g, "draw_by_repetition", False) else 0.0))
    eng.quit()

    rows = np.array(rows, dtype=float)
    cp = rows[:, 0]; ts = rows[:, 1]; pW, pD, pL = rows[:, 2], rows[:, 3], rows[:, 4]; drep = rows[:, 5]
    bins = [(-1e9,-3,"losing<-3"),(-3,-1,"-3..-1"),(-1,1,"equal"),(1,3,"+1..+3"),(3,1e9,"WIN>+3")]
    print(f"{'SF bin':>12} {'n':>4} {'mean target_V':>14} {'mean P(W)':>10} {'mean P(D)':>10} "
          f"{'mean P(L)':>10} {'%rep-game':>10}")
    for lo, hi, lab in bins:
        m = (cp > lo) & (cp <= hi)
        if m.sum() == 0:
            continue
        print(f"{lab:>12} {int(m.sum()):>4} {ts[m].mean():>+14.3f} {pW[m].mean():>10.3f} "
              f"{pD[m].mean():>10.3f} {pL[m].mean():>10.3f} {drep[m].mean()*100:>9.0f}%")
    print(f"\noverall corr(SF eval, ACTUAL target_V) = {np.corrcoef(cp, ts)[0,1]:+.3f}")
    # CRITICAL: among WINNING (>+3) positions, how many got a NEGATIVE target (tilted to loss)?
    winm = cp > 3
    if winm.sum():
        neg = np.mean(ts[winm] < -0.05)
        print(f"WINNING>+3 positions with target_V < -0.05 (taught as LOSS): {neg*100:.0f}% "
              f"(n={int(winm.sum())})")
        print(f"  of those, in rep-draw games: {np.mean(drep[winm]>0.5)*100:.0f}%")


if __name__ == "__main__":
    main()
