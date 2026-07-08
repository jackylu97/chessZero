"""Why does 200-sim numpy MCTS only VISIT ~7 of ~31 legal root children?

Dump, for a handful of real root positions:
  - full prior distribution over legal moves (entropy, max, top-k mass)
  - PUCT prior_score vs value_score magnitudes at sim 1 and at the end
  - the visit histogram

Compares to what AlphaZero/MuZero reference would do (Dirichlet at root + many
children visited). Also runs WITH dirichlet noise (as self-play does) to see if
that changes coverage.
"""
import argparse
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer
from src.mcts.mcts import MCTS
from scripts.eval_checkpoint_health import build_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.pt")
    ap.add_argument("--buf", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    dev = args.device
    buf_path = args.buf or (os.path.splitext(args.checkpoint)[0] + ".buf")
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    HF = cfg.history_frames
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    buf = ReplayBuffer(max_size=10000); buf.load(buf_path, game=game)
    rng = np.random.default_rng(args.seed)

    games = [g for g in buf.buffer if len(g.actions) >= 12]
    cfg.num_simulations = args.sims
    mcts = MCTS(net, game, cfg, dev)

    print(f"sims={args.sims}  PB_C_BASE={MCTS.PB_C_BASE}  PB_C_INIT={MCTS.PB_C_INIT}  "
          f"draw_score={cfg.draw_score}")
    print("=" * 100)

    for trial in range(args.n):
        g = games[int(rng.integers(len(games)))]
        ply = int(rng.integers(0, len(g.actions)))
        obs = g._stack_history(ply, HF)
        legal = g.legal_actions_list[ply]

        # --- prior distribution at root (no noise) ---
        with torch.no_grad():
            hidden, policy_logits, value = net.initial_inference(obs.unsqueeze(0).to(dev))
        pl = policy_logits.squeeze(0)
        legal_idx = torch.tensor(legal, dtype=torch.long)
        masked = torch.full_like(pl, float("-inf")); masked[legal_idx] = pl[legal_idx]
        priors = torch.softmax(masked, 0)[legal_idx].cpu().numpy()
        priors_sorted = np.sort(priors)[::-1]
        ent = float(-(priors * np.log(priors + 1e-12)).sum())
        max_ent = math.log(len(legal))

        for noise in (False, True):
            root = mcts.run(obs, legal, add_noise=noise)
            visits = root.child_visits
            nvis = int((visits > 0).sum())
            vsorted = np.sort(visits)[::-1]
            # effective number of moves visited (perplexity of visit dist)
            vp = visits / visits.sum()
            vp = vp[vp > 0]
            visit_perp = float(np.exp(-(vp * np.log(vp)).sum()))
            tag = "NOISE" if noise else "clean"
            if not noise:
                print(f"\n[trial {trial}] legal={len(legal)} priorEntropy={ent:.3f}/{max_ent:.3f} "
                      f"prior_top5={np.round(priors_sorted[:5],3)} prior_max={priors.max():.3f}")
            print(f"   [{tag}] visited={nvis:2d}/{len(legal)} visit_perplexity={visit_perp:.2f} "
                  f"top6_visits={vsorted[:6].astype(int)}  rootV={root.value:+.3f}")

        # PUCT magnitude analysis at sim-1 conditions (root just expanded, all visits=0)
        # prior_score = pb_c * prior * sqrt(N)/(1+0); at N=0, sqrt(N)=0 -> ALL prior_scores 0!
        # At N=1 (after first sim): pb_c*prior*1/(1+v)
        pb_c0 = math.log((0 + MCTS.PB_C_BASE + 1) / MCTS.PB_C_BASE) + MCTS.PB_C_INIT
        pb_c_late = math.log((args.sims + MCTS.PB_C_BASE + 1) / MCTS.PB_C_BASE) + MCTS.PB_C_INIT
        # at root, value_score for unvisited children is 0 (visits>0 gate); only the
        # FIRST-visited child gets a value_score. So selection is prior-dominated early.
        print(f"   pb_c(N=0)={pb_c0:.4f} pb_c(N={args.sims})={pb_c_late:.4f}  "
              f"prior_score(top,N=1,sqrtN=1)={pb_c0*priors.max():.4f}")


if __name__ == "__main__":
    main()
