"""Fast (no-forward) sanity check of value TARGETS the value head is trained on.

Reads the real .buf, builds make_target outputs exactly as sample_batch does,
and checks:
  1. WDL target POV alternation within decisive games (STM one-hot flips per ply).
  2. Distribution of root-ply value targets across the buffer (is the head being
     given a non-degenerate target, or is everything ~draw?).
  3. Decisive-stratum vs full-buffer target std (decisive_sample_frac=0.5 path).
  4. Whether root_values stored in self-play games carry within-position signal
     (the thing MCTS needs but the prior probes said is ~0.008 std).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

CKPT_DIR = "checkpoints/chess/2026_06_19_cold2_pc"


def wdl_to_scalar_np(w):
    return w[..., 0] - w[..., 2]


def main():
    game = ChessGame()
    cfg = get_config("chess_small")
    torch.serialization.add_safe_globals([MuZeroConfig])

    for step in [1000, 6000, 30000]:
        rb = ReplayBuffer(cfg.replay_buffer_size)
        rb.load(os.path.join(CKPT_DIR, f"checkpoint_{step}.buf"), game=game)
        games = rb.buffer
        n_dec = sum(1 for g in games if abs(g.game_outcome) >= 0.5)
        n_draw = sum(1 for g in games if g.game_outcome == 0.0)
        n_ext = sum(1 for g in games if g.external_values)
        print(f"\n{'='*70}\nCHECKPOINT {step}: {len(games)} games | decisive={n_dec} draw={n_draw} warmstart={n_ext}")

        # 1. POV alternation: pick a decisive game, dump WDL root targets per ply.
        dec_games = [g for g in games if abs(g.game_outcome) >= 0.5]
        if dec_games:
            g = dec_games[0]
            print(f"  [POV check] decisive game outcome={g.game_outcome:+.0f} len={len(g)}")
            wrong = 0
            for ply in range(min(len(g), 8)):
                obs_list, mask, acts, vals, rew, pol = g.make_target(
                    ply, 0, cfg.td_steps, cfg.discount, game.action_space_size,
                    value_head_type="wdl", history_frames=cfg.history_frames,
                    eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
                    q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
                    selfplay_q_ratio=cfg.selfplay_q_ratio,
                    repetition_penalty=cfg.repetition_penalty,
                    repetition_penalty_window=int(cfg.repetition_penalty_window),
                    repetition_penalty_decay=float(cfg.repetition_penalty_decay))
                w = np.asarray(vals[0], dtype=np.float32)
                stm_white = (ply % 2 == 0)
                stm_won = (g.game_outcome > 0) == stm_white
                expect = "W" if stm_won else "L"
                got = "W" if w[0] > w[2] else ("L" if w[2] > w[0] else "D")
                if got != expect:
                    wrong += 1
                print(f"    ply {ply} stm={'W' if stm_white else 'B'} WDL=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}] "
                      f"scalar={wdl_to_scalar_np(w):+.2f} expect={expect} got={got}")
            print(f"  [POV check] {wrong} mismatches in first 8 plies (should be 0)")

        # 2/3. Sample 2000 random root-ply WDL targets (full buffer) and the decisive stratum.
        rng = np.random.default_rng(0)
        def sample_targets(pool, n=2000):
            scal = []
            for _ in range(n):
                g = pool[rng.integers(len(pool))]
                ply = int(rng.integers(len(g)))
                _, _, _, vals, _, _ = g.make_target(
                    ply, 0, cfg.td_steps, cfg.discount, game.action_space_size,
                    value_head_type="wdl", history_frames=1,
                    eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
                    q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
                    selfplay_q_ratio=cfg.selfplay_q_ratio,
                    repetition_penalty=cfg.repetition_penalty,
                    repetition_penalty_window=int(cfg.repetition_penalty_window),
                    repetition_penalty_decay=float(cfg.repetition_penalty_decay))
                scal.append(wdl_to_scalar_np(np.asarray(vals[0], dtype=np.float32)))
            return np.array(scal)
        full = sample_targets(games)
        print(f"  [targets full ] root scalar V: mean={full.mean():+.3f} std={full.std():.3f} "
              f"frac|V|>0.5={np.mean(np.abs(full)>0.5):.2f} frac~0={np.mean(np.abs(full)<0.05):.2f}")
        if dec_games:
            dec = sample_targets(dec_games)
            print(f"  [targets decis] root scalar V: mean={dec.mean():+.3f} std={dec.std():.3f} "
                  f"frac|V|>0.5={np.mean(np.abs(dec)>0.5):.2f} frac~0={np.mean(np.abs(dec)<0.05):.2f}")

        # 4. Within-position root_value signal: std of stored root_values across plies
        #    of the SAME game (the MCTS root V the prior probes said is ~flat).
        rv_within = []
        for g in games[:200]:
            if len(g.root_values) >= 4:
                rv_within.append(float(np.std(g.root_values)))
        if rv_within:
            print(f"  [root_values ] within-game std (mean over {len(rv_within)} games)={np.mean(rv_within):.4f}  "
                  f"(stored MCTS root V variation across plies)")


if __name__ == "__main__":
    main()
