"""How many distinct moves does the STORED self-play policy target actually
spread visits over, and how does it compare to the prior / legal count?

If self-play (BatchedMCTS, sample_k=50, 200 sims, +Dirichlet) produced policy
targets with support over only ~6-9 moves, the policy is being trained on very
sharp targets -> low diversity. Also: does sample_k actually expand more than
the numpy engine's full-action leaf? Reports stored-target support, entropy,
and argmax-mass distribution across the whole buffer.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf")
    args = ap.parse_args()
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame()
    buf = ReplayBuffer(max_size=10000); buf.load(args.buf, game=game)
    print(f"{args.buf}: {len(buf.buffer)} games")

    supports, ents, argmax_mass, legal_counts, second_mass = [], [], [], [], []
    n_steps = 0
    for g in buf.buffer:
        for i, p in enumerate(g.policies):
            if p is None or p.sum() <= 0:
                continue
            nz = p[p > 0]
            supports.append(len(nz))
            ents.append(float(-(nz * np.log(nz + 1e-12)).sum()))
            ps = np.sort(nz)[::-1]
            argmax_mass.append(float(ps[0]))
            second_mass.append(float(ps[1]) if len(ps) > 1 else 0.0)
            if i < len(g.legal_actions_list):
                legal_counts.append(len(g.legal_actions_list[i]))
            n_steps += 1
            if n_steps >= 200000:
                break
        if n_steps >= 200000:
            break

    def s(name, v, fmt="{:.3f}"):
        v = np.array(v, dtype=np.float64)
        print(f"  {name:30s} mean={fmt.format(v.mean())} median={fmt.format(np.median(v))} "
              f"p10={fmt.format(np.percentile(v,10))} p90={fmt.format(np.percentile(v,90))} "
              f"min={fmt.format(v.min())} max={fmt.format(v.max())}")

    print(f"\nstored policy-target stats over {n_steps} plies:")
    s("support (#nonzero moves)", supports, "{:.2f}")
    s("entropy (nats)", ents)
    s("argmax mass (top move prob)", argmax_mass)
    s("second move prob", second_mass)
    s("legal move count", legal_counts, "{:.1f}")

    sup = np.array(supports)
    print(f"\nsupport histogram: ==1:{(sup==1).mean()*100:.1f}%  "
          f"<=2:{(sup<=2).mean()*100:.1f}%  <=3:{(sup<=3).mean()*100:.1f}%  "
          f"<=5:{(sup<=5).mean()*100:.1f}%  <=10:{(sup<=10).mean()*100:.1f}%  "
          f">10:{(sup>10).mean()*100:.1f}%")
    am = np.array(argmax_mass)
    print(f"argmax-mass: >0.9:{(am>0.9).mean()*100:.1f}%  >0.7:{(am>0.7).mean()*100:.1f}%  "
          f">0.5:{(am>0.5).mean()*100:.1f}%  <0.3:{(am<0.3).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
