"""Cheap probe (no MCTS): is the STORED policy target internally sane?

For many stored self-play plies, check:
  - policy sums to ~1
  - all policy mass is on LEGAL actions (no illegal leakage)
  - the policy's support size vs #legal (how concentrated)
  - is the played action (g.actions[ply]) in the policy support / is it the argmax?
  - does the stored root_value sit in [-1, +1] and cluster near draw_score?
  - entropy of the policy target across plies/phases
This catches target mis-allocation / misalignment WITHOUT re-running search.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory


def load_games(buf_path, game, max_games=400):
    import pickle
    games = []
    with open(buf_path, "rb") as f:
        first = pickle.load(f)
        version = first["version"]; n = first["n_records"]
        for _ in range(n):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if len(games) >= max_games:
                break
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/chess/2026_06_19_cold2_pc")
    ap.add_argument("--buf", default=None)
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    buf_path = args.buf or os.path.join(args.ckpt_dir, "checkpoint_30000.buf")
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    games = load_games(buf_path, game, args.max_games)
    sp = [g for g in games if not g.external_values and len(g.actions) >= 6]
    print(f"{buf_path}: {len(games)} games, {len(sp)} self-play")

    sums = []; illeg = []; supp = []; nlegal = []
    played_in_supp = 0; played_is_argmax = 0; total_plies = 0
    rv = []; ent = []; rv_at_played = []
    argmax_eq_played_byphase = {"open": [0, 0], "mid": [0, 0], "late": [0, 0]}

    rng = np.random.default_rng(args.seed)
    for g in sp:
        L = len(g.actions)
        for ply in range(L):
            if ply >= len(g.policies):
                continue
            p = g.policies[ply].astype(np.float64)
            legal = set(int(a) for a in g.legal_actions_list[ply])
            s = float(p.sum())
            sums.append(s)
            illeg_mass = float(sum(p[a] for a in range(len(p)) if a not in legal))
            illeg.append(illeg_mass)
            nz = int(np.count_nonzero(p))
            supp.append(nz)
            nlegal.append(len(legal))
            played = int(g.actions[ply])
            total_plies += 1
            if p[played] > 0:
                played_in_supp += 1
            am = int(np.argmax(p))
            if am == played:
                played_is_argmax += 1
            # entropy
            pp = p[p > 0]
            ent.append(float(-(pp * np.log(pp)).sum()) if len(pp) else 0.0)
            if ply < len(g.root_values):
                rv.append(float(g.root_values[ply]))
            # phase bucket
            frac = ply / max(L - 1, 1)
            bucket = "open" if frac < 0.25 else ("late" if frac > 0.75 else "mid")
            argmax_eq_played_byphase[bucket][0] += int(am == played)
            argmax_eq_played_byphase[bucket][1] += 1

    def st(name, arr, fmt="{:.4f}"):
        a = np.array(arr, dtype=np.float64)
        print(f"  {name:34s} mean={fmt.format(a.mean())} med={fmt.format(np.median(a))} "
              f"min={fmt.format(a.min())} max={fmt.format(a.max())} (n={len(a)})")

    print(f"\ntotal plies analyzed: {total_plies}")
    st("policy sum", sums)
    st("ILLEGAL mass in policy", illeg, "{:.6f}")
    st("policy support size (#nonzero)", supp, "{:.1f}")
    st("#legal actions", nlegal, "{:.1f}")
    st("stored root_value", rv, "{:+.4f}")
    st("policy entropy (nats)", ent)
    print(f"  played action in policy support: {played_in_supp}/{total_plies} "
          f"({100*played_in_supp/total_plies:.1f}%)")
    print(f"  played action == policy argmax:  {played_is_argmax}/{total_plies} "
          f"({100*played_is_argmax/total_plies:.1f}%)")
    for b, (hit, tot) in argmax_eq_played_byphase.items():
        if tot:
            print(f"    [{b:4s}] argmax==played: {hit}/{tot} ({100*hit/tot:.1f}%)")
    print("\n  policy sum != 1 OR illegal mass > 0 => target mis-allocation BUG.")
    print("  played != argmax is EXPECTED under temperature sampling (T=1 early).")


if __name__ == "__main__":
    main()
