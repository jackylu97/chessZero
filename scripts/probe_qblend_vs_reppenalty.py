"""Quantify the TUG-OF-WAR at rep plies: repetition_penalty pushes target toward
LOSS, but selfplay_q_ratio blends the (phantom-win) root_value back toward WIN.
Net effect on the value target? CPU."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory
from src.model.utils import eval_to_wdl, WDL_W, WDL_D, WDL_L


def load_buffer_games(path, game, limit=None):
    games = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        version = header["version"]
        for i in range(header["n_records"]):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if limit and len(games) >= limit:
                break
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    games = load_buffer_games(args.buf, game, args.limit)
    rep = [g for g in games if g.draw_by_repetition]

    q = cfg.selfplay_q_ratio
    rd = cfg.repetition_penalty
    dec = cfg.repetition_penalty_decay
    a, b = cfg.eval_to_wdl_alpha, cfg.eval_to_wdl_beta

    # At each rep ply in the LAST 8, compute:
    #   legacy_with_penalty = [0, 1-d, d]   (d = rd * dec^plies_to_end)
    #   blended = (1-q)*legacy + q*eval_to_wdl(root_value)
    #   compare scalar(legacy) vs scalar(blended): does the q-blend ERASE the penalty?
    sc_legacy, sc_blended, dvals, rootvs = [], [], [], []
    # also the L-component: penalty raises L, q-blend may raise W instead.
    Lp_legacy, Lp_blended = [], []
    Wp_blended = []
    for g in rep:
        n = len(g.actions)
        rep_end = len(g) - 1
        for ply in range(max(0, n - 8), n):
            plies_to_end = rep_end - ply
            d = rd * (dec ** plies_to_end)
            legacy = np.array([0.0, 1.0 - d, d])
            rv = g.root_values[ply] if ply < len(g.root_values) else 0.0
            pw, pd, pl = eval_to_wdl(rv, alpha=a, beta=b)
            blended_in = np.array([pw, pd, pl])
            blended = (1.0 - q) * legacy + q * blended_in
            sc_legacy.append(legacy[WDL_W] - legacy[WDL_L])
            sc_blended.append(blended[WDL_W] - blended[WDL_L])
            Lp_legacy.append(legacy[WDL_L]); Lp_blended.append(blended[WDL_L])
            Wp_blended.append(blended[WDL_W])
            dvals.append(d); rootvs.append(rv)
    sc_legacy = np.array(sc_legacy); sc_blended = np.array(sc_blended)
    Lp_legacy = np.array(Lp_legacy); Lp_blended = np.array(Lp_blended)
    Wp_blended = np.array(Wp_blended); dvals = np.array(dvals); rootvs = np.array(rootvs)
    print(f"q_self={q} rep_penalty={rd} decay={dec}")
    print(f"\nAt last-8 plies of rep-draws ({len(sc_legacy)} plies):")
    print(f"  penalty-only target scalar (W-L): mean={sc_legacy.mean():+.4f}  (should be NEGATIVE, anti-draw)")
    print(f"  after q-blend target scalar (W-L): mean={sc_blended.mean():+.4f}")
    print(f"  => q-blend shifts the anti-draw scalar by {sc_blended.mean()-sc_legacy.mean():+.4f}")
    print(f"\n  L-prob: penalty-only mean={Lp_legacy.mean():.4f}  after-blend mean={Lp_blended.mean():.4f}")
    print(f"  W-prob: after-blend mean={Wp_blended.mean():.4f}  (penalty-only W=0)")
    print(f"  mean d (penalty mass)={dvals.mean():.4f}  mean root_value={rootvs.mean():+.4f}")
    # How often does the q-blend make the target NET POSITIVE (win) despite the rep penalty?
    print(f"\n  frac of rep plies where blended W-L > 0 (target says WIN at a draw!): {np.mean(sc_blended>0.01):.3f}")
    print(f"  frac where blended W > blended L: {np.mean(Wp_blended>Lp_blended):.3f}")
    # subset: plies where root_value is high (phantom win)
    hi = rootvs > 0.5
    if hi.any():
        print(f"\n  Subset root_value>0.5 (phantom win) — {hi.sum()} plies:")
        print(f"    penalty-only scalar mean={sc_legacy[hi].mean():+.4f}  blended scalar mean={sc_blended[hi].mean():+.4f}")
        print(f"    => at phantom-win rep plies the target says WIN={np.mean(sc_blended[hi]>0):.3f} of the time")


if __name__ == "__main__":
    main()
