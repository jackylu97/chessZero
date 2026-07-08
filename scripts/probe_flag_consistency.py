"""Check flag/outcome consistency + the actual aggregate value-target signal
the trainer sees. Cross-check vs probe_value_target_audit. CPU only."""
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

    # 1. Flag/outcome consistency.
    bad = [g for g in games if g.draw_by_repetition and g.game_outcome != 0.0]
    bad2 = [g for g in games if g.draw_by_no_progress and g.game_outcome != 0.0]
    print(f"games with draw_by_repetition AND outcome!=0: {len(bad)}")
    print(f"games with draw_by_no_progress AND outcome!=0: {len(bad2)}")
    # decisive games with rep flag would be a real bug.

    # 2. Aggregate value-target signal the trainer sees: simulate sample_batch's
    #    make_target over MANY random (game, ply) draws, compute the scalar of the
    #    WDL target, and bin by what the network's stored root_value says.
    from src.model.utils import WDL_W, WDL_D, WDL_L
    rng = np.random.default_rng(0)
    n_samples = 20000
    # Mimic decisive_sample_frac=0.5 stratified sampling.
    dec_idx = [i for i, g in enumerate(games) if abs(g.game_outcome) >= 0.5]
    draw_idx = [i for i, g in enumerate(games) if abs(g.game_outcome) < 0.5]
    print(f"\nstratify: decisive games={len(dec_idx)} draw games={len(draw_idx)}")

    def scalar_of(tgt, ds):
        return float(tgt[WDL_W] - tgt[WDL_L] + ds * tgt[WDL_D])

    # FLAT sampling (no stratification) — what fraction of training value-target
    # mass is draw vs decisive?
    for mode, idx_pool, frac in [("FLAT", list(range(len(games))), None),
                                  ("DECISIVE-stratified(0.5)", None, 0.5)]:
        tgt_scalars = []
        is_draw_tgt = 0
        for s in range(n_samples):
            if frac is None:
                gi = idx_pool[rng.integers(len(idx_pool))]
            else:
                if rng.random() < frac and dec_idx:
                    gi = dec_idx[rng.integers(len(dec_idx))]
                else:
                    gi = draw_idx[rng.integers(len(draw_idx))] if draw_idx else dec_idx[rng.integers(len(dec_idx))]
            g = games[gi]
            ply = rng.integers(len(g))
            _, _, _, values, _, _ = g.make_target(
                ply, 0, cfg.td_steps, cfg.discount, game.action_space_size,
                value_head_type=cfg.value_head_type, history_frames=1,
                eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
                q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
                selfplay_q_ratio=cfg.selfplay_q_ratio,
                repetition_penalty=cfg.repetition_penalty,
                repetition_penalty_window=cfg.repetition_penalty_window,
                repetition_penalty_decay=cfg.repetition_penalty_decay)
            tgt = values[0]
            tgt_scalars.append(scalar_of(tgt, 0.0))
            if tgt[WDL_D] > 0.5:
                is_draw_tgt += 1
        ts = np.array(tgt_scalars)
        print(f"\n[{mode}] over {n_samples} (game,ply) draws:")
        print(f"   target scalar(ds=0): mean={ts.mean():+.4f} std={ts.std():.4f} "
              f"min={ts.min():+.3f} max={ts.max():+.3f}")
        print(f"   frac with P(D)>0.5 (drawish target): {is_draw_tgt/n_samples:.3f}")
        print(f"   frac |scalar|>0.5 (decisive-ish target): {(np.abs(ts)>0.5).mean():.3f}")


if __name__ == "__main__":
    main()
