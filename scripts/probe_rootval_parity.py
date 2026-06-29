"""Hunt the alternating root_value pattern: is there a systematic White/Black
asymmetry in stored self-play root_values (perspective bug) or is it just noise?
Also: how do stored root_values relate to outcome? CPU only."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory


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
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    games = load_buffer_games(args.buf, game, args.limit)

    # 1. White-ply vs Black-ply stored root_value, over ALL games + rep-draws only.
    for label, pool in [("ALL games", games),
                        ("REP-DRAWS only", [g for g in games if g.draw_by_repetition]),
                        ("DECISIVE only", [g for g in games if g.game_outcome != 0.0])]:
        rv_white, rv_black = [], []
        for g in pool:
            for ply, rv in enumerate(g.root_values):
                if ply % 2 == 0:
                    rv_white.append(rv)
                else:
                    rv_black.append(rv)
        rw = np.array(rv_white); rb = np.array(rv_black)
        print(f"\n[{label}] n_games={len(pool)}")
        print(f"   White-to-move root_value: mean={rw.mean():+.4f} std={rw.std():.4f} "
              f"frac>0.3={np.mean(rw>0.3):.3f} frac<-0.3={np.mean(rw<-0.3):.3f}  (n={len(rw)})")
        print(f"   Black-to-move root_value: mean={rb.mean():+.4f} std={rb.std():.4f} "
              f"frac>0.3={np.mean(rb>0.3):.3f} frac<-0.3={np.mean(rb<-0.3):.3f}  (n={len(rb)})")

    # 2. In rep-draws, look at the LAST 8 plies: alternating high/low?
    print(f"\n--- Last-8-ply root_value pattern in rep-draws (first 6 games) ---")
    rep = [g for g in games if g.draw_by_repetition]
    for g in rep[:6]:
        n = len(g.root_values)
        tail = [(ply, "W" if ply % 2 == 0 else "B", g.root_values[ply])
                for ply in range(max(0, n - 8), n)]
        s = "  ".join(f"{p}{t}:{v:+.2f}" for p, t, v in tail)
        print(f"  len={len(g)} : {s}")

    # 3. Correlation of stored root_value with sign*outcome at each ply (decisive games):
    #    if root_value tracked the outcome, decisive games would show strong corr.
    rvs, zs = [], []
    for g in games:
        if g.game_outcome == 0.0:
            continue
        for ply, rv in enumerate(g.root_values):
            stm_is_white = (ply % 2 == 0)
            stm_won = (g.game_outcome > 0.0) == stm_is_white
            z = 1.0 if stm_won else -1.0
            rvs.append(rv); zs.append(z)
    rvs = np.array(rvs); zs = np.array(zs)
    if rvs.std() > 1e-9:
        print(f"\ncorr(stored root_value, sign*outcome) on decisive games: "
              f"{np.corrcoef(rvs, zs)[0,1]:+.3f}  (n={len(rvs)})")
        print(f"   mean root_value when STM eventually WINS:  {rvs[zs>0].mean():+.4f}")
        print(f"   mean root_value when STM eventually LOSES: {rvs[zs<0].mean():+.4f}")


if __name__ == "__main__":
    main()
