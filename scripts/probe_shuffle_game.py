"""Dump per-ply targets + root_values for a few real threefold-draw games, to see
what signal the value/policy heads get and whether the repetition_penalty + the
visit-count policy target create a 'reach repetition faster' incentive.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer
torch.serialization.add_safe_globals([MuZeroConfig])

CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"


def main():
    game = ChessGame(); cfg = get_config("chess_small")
    rb = ReplayBuffer(cfg); rb.load(BUF, game=game)
    draws = [g for g in rb.buffer if g.game_outcome == 0 and getattr(g, "draw_by_repetition", False)
             and not g.external_values]
    print(f"{len(draws)} threefold-draw games")
    lens = np.array([len(g.actions) for g in draws])
    # Look at a SHORT threefold draw (the collapse symptom) and a longer one.
    order = np.argsort(lens)
    for tag, gi in [("shortest", order[0]), ("short-p10", order[len(order)//10]),
                    ("median", order[len(order)//2])]:
        gh = draws[int(gi)]
        L = len(gh.actions)
        print(f"\n=== {tag} threefold draw: {L} plies, outcome={gh.game_outcome} ===")
        # Per-ply policy target concentration (max visit fraction) + root value.
        rv = np.array(gh.root_values)
        # policy targets: sparse arrays; compute max prob per ply
        maxp = []
        for p in gh.policies:
            p = np.asarray(p)
            maxp.append(float(p.max()) if p.size else 0.0)
        maxp = np.array(maxp)
        print(f"  root_value: mean={rv.mean():+.3f} std={rv.std():.3f} "
              f"first5={np.round(rv[:5],2)} last5={np.round(rv[-5:],2)}")
        print(f"  policy max-prob (target sharpness): mean={maxp.mean():.3f} "
              f"first5={np.round(maxp[:5],2)} last5={np.round(maxp[-5:],2)}")
        # WDL target at a few plies (early, mid, near-end) with repetition_penalty.
        for frac in [0.1, 0.5, 0.9]:
            si = max(0, min(L - 1, int(frac * L)))
            out = gh.make_target(si, 0, cfg.td_steps, cfg.discount, game.action_space_size,
                                 history_frames=cfg.history_frames, value_head_type="wdl",
                                 q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
                                 selfplay_q_ratio=cfg.selfplay_q_ratio,
                                 repetition_penalty=cfg.repetition_penalty,
                                 repetition_penalty_decay=cfg.repetition_penalty_decay)
            tv = out[3][0]
            print(f"    ply {si:4d} ({frac:.0%}): WDL target=({tv[0]:.3f},{tv[1]:.3f},{tv[2]:.3f})  "
                  f"plies_to_end={L-1-si}")


if __name__ == "__main__":
    main()
