"""Trace make_target / _wdl_target_at on real cold2 buffer games to hunt for a
draw-collapse BUG vs the fundamental V^pi=draw target issue. CPU only."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory


def load_buffer_games(path, game, limit=None):
    games = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        assert isinstance(header, dict) and header.get("version") in (2, 3), header
        version = header["version"]
        n = header["n_records"]
        for i in range(n):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if limit and len(games) >= limit:
                break
    return games, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = get_config("chess_small")
    cfg.device = "cpu"
    game = ChessGame()
    print(f"value_head_type={cfg.value_head_type} draw_score={cfg.draw_score} "
          f"selfplay_q_ratio={cfg.selfplay_q_ratio} warmstart_q_ratio={cfg.warmstart_q_ratio} "
          f"q_ratio={cfg.q_ratio} td_steps={cfg.td_steps} discount={cfg.discount}")
    print(f"repetition_penalty={cfg.repetition_penalty} decay={cfg.repetition_penalty_decay} "
          f"window={cfg.repetition_penalty_window} eval_to_wdl a/b={cfg.eval_to_wdl_alpha}/{cfg.eval_to_wdl_beta}")

    games, header = load_buffer_games(args.buf, game, args.limit)
    print(f"\nLoaded {len(games)} games from {args.buf} (header total_games={header.get('total_games')})")

    # Characterize the buffer.
    n_draw = sum(1 for g in games if g.game_outcome == 0.0)
    n_decisive = sum(1 for g in games if g.game_outcome != 0.0)
    n_rep = sum(1 for g in games if g.draw_by_repetition)
    n_nop = sum(1 for g in games if g.draw_by_no_progress)
    n_ext = sum(1 for g in games if g.external_values)
    lens = np.array([len(g) for g in games])
    print(f"\nBUFFER COMPOSITION:")
    print(f"  draws={n_draw} decisive={n_decisive}  by_repetition={n_rep} by_no_progress={n_nop}  with_external_values={n_ext}")
    print(f"  game length: min={lens.min()} max={lens.max()} mean={lens.mean():.1f} median={np.median(lens):.0f}")
    outcomes = np.array([g.game_outcome for g in games])
    print(f"  game_outcome values: {sorted(set(outcomes.tolist()))}")
    rep_lens = np.array([len(g) for g in games if g.draw_by_repetition])
    if len(rep_lens):
        print(f"  repetition-game length: min={rep_lens.min()} max={rep_lens.max()} mean={rep_lens.mean():.1f}")
    dec_games = [g for g in games if g.game_outcome != 0.0]
    if dec_games:
        dl = np.array([len(g) for g in dec_games])
        print(f"  decisive-game length: min={dl.min()} max={dl.max()} mean={dl.mean():.1f}")

    return games, cfg, game


if __name__ == "__main__":
    main()
