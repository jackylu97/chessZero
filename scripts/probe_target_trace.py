"""Detailed per-ply target trace for cold2 buffer games. CPU only."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory
from src.model.utils import eval_to_wdl


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


def trace_game(g, cfg, game, label, last_n=14):
    print(f"\n{'='*90}\n{label}: len={len(g)} outcome={g.game_outcome} "
          f"by_rep={g.draw_by_repetition} by_nop={g.draw_by_no_progress} "
          f"n_root_values={len(g.root_values)} n_policies={len(g.policies)} n_external={len(g.external_values)}")
    n = len(g)
    # Show last_n plies (where repetition happens) and a few early ones.
    plies_to_show = list(range(max(0, n - last_n), n)) + [0, n // 4, n // 2]
    plies_to_show = sorted(set(plies_to_show))
    print(f"  ply  stm   rootV    -> make_target WDL [W,D,L]   sign*outcome  rep_weight  delta")
    rep_end_idx = n - 1
    for ply in plies_to_show:
        # Call make_target at this ply with K=0 to get just this ply's value target.
        obs_list, obs_mask, actions, values, rewards, policies = g.make_target(
            ply, 0, cfg.td_steps, cfg.discount, game.action_space_size,
            value_head_type=cfg.value_head_type,
            history_frames=1,  # don't need full history for value target
            eval_to_wdl_alpha=cfg.eval_to_wdl_alpha,
            eval_to_wdl_beta=cfg.eval_to_wdl_beta,
            q_ratio=cfg.q_ratio,
            warmstart_q_ratio=cfg.warmstart_q_ratio,
            selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,
            repetition_penalty_window=cfg.repetition_penalty_window,
            repetition_penalty_decay=cfg.repetition_penalty_decay,
        )
        tgt = values[0]
        stm = "W" if ply % 2 == 0 else "B"
        rv = g.root_values[ply] if ply < len(g.root_values) else float("nan")
        # outcome one-hot sign
        if g.game_outcome == 0.0:
            so = 0.0
        else:
            stm_is_white = (ply % 2 == 0)
            stm_won = (g.game_outcome > 0.0) == stm_is_white
            so = 1.0 if stm_won else -1.0
        # rep weight
        if cfg.repetition_penalty_decay > 0:
            w = cfg.repetition_penalty_decay ** (rep_end_idx - ply)
        else:
            w = 1.0
        d = cfg.repetition_penalty * w if (g.game_outcome == 0.0 and (g.draw_by_repetition or g.draw_by_no_progress)) else 0.0
        marker = "  <-LAST" if ply == n - 1 else ""
        print(f"  {ply:4d}  {stm}   {rv:+.3f}  -> [{tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}]   "
              f"{so:+.1f}        {w:.4f}    {d:.4f}{marker}")
    # scalar value of target, and mean over game
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()

    cfg = get_config("chess_small")
    cfg.device = "cpu"
    game = ChessGame()
    games = load_buffer_games(args.buf, game, args.limit)

    rep_games = [g for g in games if g.draw_by_repetition]
    dec_games = [g for g in games if g.game_outcome != 0.0]

    # Trace 2 repetition draws and 2 decisive games.
    for i, g in enumerate(rep_games[:2]):
        trace_game(g, cfg, game, f"REP-DRAW #{i}")
    for i, g in enumerate(dec_games[:2]):
        trace_game(g, cfg, game, f"DECISIVE #{i} (outcome={g.game_outcome})")


if __name__ == "__main__":
    main()
