"""Fast: stored MCTS policy at ply 0 across self-play games — is the OPENING
policy itself collapsed (near-deterministic) at 30k vs 6k?"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.config import MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory

def load(buf, game, mx):
    gs = []
    with open(buf, "rb") as f:
        first = pickle.load(f); ver = first["version"]; n = first["n_records"]
        for _ in range(n):
            rec, _ = pickle.load(f)
            if ver == 3:
                rec = GameHistory.from_compact_dict(rec, game)
            gs.append(rec)
            if len(gs) >= mx: break
    return gs

game = ChessGame(); torch.serialization.add_safe_globals([MuZeroConfig])
for step in (6000, 30000):
    buf = f"checkpoints/chess/2026_06_19_cold2_pc/checkpoint_{step}.buf"
    gs = [g for g in load(buf, game, 300) if not g.external_values and g.policies]
    p0 = np.stack([g.policies[0] for g in gs])           # [G, A]
    ent = []
    for p in p0:
        pp = p[p > 0]
        ent.append(float(-(pp*np.log(pp)).sum()) if len(pp) else 0.0)
    mean_p0 = p0.mean(0)
    top_actions = np.argsort(-mean_p0)[:5]
    # how peaked is the AVERAGE ply-0 policy
    print(f"\nstep {step}: {len(gs)} games")
    print(f"  ply-0 policy entropy: mean={np.mean(ent):.3f} med={np.median(ent):.3f} "
          f"min={min(ent):.3f} max={max(ent):.3f}")
    print(f"  per-game ply-0 max-prob: mean={p0.max(1).mean():.3f} "
          f"med={np.median(p0.max(1)):.3f}")
    print(f"  mean ply-0 policy top-5 actions/mass: "
          f"{[(int(a), round(float(mean_p0[a]),3)) for a in top_actions]}")
    print(f"  fraction of games whose ply-0 argmax == global top action {int(top_actions[0])}: "
          f"{np.mean(p0.argmax(1)==top_actions[0]):.3f}")
