#!/usr/bin/env python3
"""Measure how INFORMATIVE the representation network's latents are.

The representation h = representation(obs) is shaped only by downstream task
gradients (root value+policy CE, plus attenuated unroll/consistency/inverse
losses) — there is no reconstruction loss. "Informative" therefore means
"linearly decodable into the quantities the heads need": position value and
game outcome. This probe freezes the net and, on a held-out set of
Stockfish-labelled positions, reports:

  - cross-position cosine  : mean pairwise cosine of latents over distinct
                             positions. ~1.0 => collapse (uninformative);
                             the position-space analogue of the dynamics
                             cross-action cosine.
  - effective rank         : entropy + participation-ratio of the latent
                             covariance spectrum. Low => dimensional collapse
                             (only a few latent dims carry signal).
  - linear-probe R2(eval)  : val R2 of a ridge map h -> Stockfish STM eval.
                             How much value information is LINEARLY accessible
                             (Alain & Bengio linear-probe methodology).
  - linear-probe R2(outcome) + decisive sign-acc : ridge map h -> STM game
                             outcome in {-1,0,+1}.

Same probe set is used for every checkpoint, so the metrics are directly
comparable across runs regardless of each run's training-data distribution.

Usage:
  .venv/bin/python scripts/probe_representation.py --checkpoint <path.pt> \
      [--shards-dir data/stockfish_injection] [--n-positions 1500] \
      [--n-games 250] [--device cpu]
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.replay_buffer import _iter_shard_games
from src.training.representation_probe import (
    dual_ridge_r2, effective_rank, mean_crosspos_cosine)


def build_network(ckpt, game, cfg, device):
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim,
        use_consistency_loss=cfg.use_consistency_loss, proj_hid=cfg.proj_hid, proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid, pred_out=cfg.pred_out, use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale, value_head_type=cfg.value_head_type,
        draw_score=cfg.draw_score, value_head_init_std=getattr(cfg, "value_head_init_std", 0.0),
        use_inverse_dynamics_loss=getattr(cfg, "use_inverse_dynamics_loss", False),
        inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 256),
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def sample_positions(shards_dir, game, hf, n_games, n_per_game, seed):
    """Return (stacked_obs [M,C,H,W], eval_stm [M], outcome_stm [M])."""
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(shards_dir, "**", "*.pkl"), recursive=True))
    rng.shuffle(paths)
    obs_list, ev_list, out_list = [], [], []
    games_used = 0
    for p in paths:
        if games_used >= n_games:
            break
        try:
            games = list(_iter_shard_games(p, game=game))
        except Exception:
            continue
        rng.shuffle(games)
        for gh in games:
            if games_used >= n_games:
                break
            n_ev = len(gh.external_values)
            if n_ev == 0:
                continue
            games_used += 1
            k = min(n_per_game, n_ev)
            plies = rng.choice(n_ev, size=k, replace=False)
            for ply in plies:
                stack = gh._stack_history(int(ply), hf)  # [C*hf, H, W], STM-relative
                obs_list.append(stack.numpy())
                ev = float(gh.external_values[int(ply)])      # already STM POV
                stm_white = (int(ply) % 2 == 0)
                out = float(gh.game_outcome) * (1.0 if stm_white else -1.0)
                ev_list.append(ev)
                out_list.append(out)
    return (np.asarray(obs_list, dtype=np.float32),
            np.asarray(ev_list, dtype=np.float32),
            np.asarray(out_list, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards-dir", default="data/stockfish_injection")
    ap.add_argument("--n-games", type=int, default=250)
    ap.add_argument("--n-per-game", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    game = ChessGame()
    cfg = get_config("chess"); cfg.device = dev
    hf = cfg.history_frames
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    step = ckpt.get("step", "?")
    net = build_network(ckpt, game, cfg, dev)

    obs, ev, out = sample_positions(args.shards_dir, game, hf, args.n_games, args.n_per_game, args.seed)
    M = len(obs)
    print(f"\n{'='*68}\nREPRESENTATION PROBE — step {step}  ({args.checkpoint})\n{'='*68}")
    print(f"positions: {M} | eval std {ev.std():.3f} | outcome dist "
          f"W/D/L = {(out>0.5).mean():.2f}/{(np.abs(out)<=0.5).mean():.2f}/{(out<-0.5).mean():.2f}")
    if M < 50:
        print("too few positions — aborting"); return

    # representations
    H = []
    with torch.no_grad():
        for s in range(0, M, 256):
            x = torch.from_numpy(obs[s:s+256]).to(dev)
            h = net.representation(x).reshape(x.shape[0], -1).float().cpu().numpy()
            H.append(h)
    H = np.concatenate(H, 0)
    D = H.shape[1]

    cpc = mean_crosspos_cosine(H, seed=args.seed)
    ent_rank, part_ratio = effective_rank(H)
    r2_eval, _ = dual_ridge_r2(H, ev, seed=args.seed)
    r2_out, op = dual_ridge_r2(H, out, seed=args.seed)
    # decisive sign accuracy from the outcome probe
    pred, yv = op
    dec = np.abs(yv) > 0.5
    sign_acc = float((np.sign(pred[dec]) == np.sign(yv[dec])).mean()) if dec.sum() else float("nan")

    print(f"latent dim D={D}")
    print(f"  cross-position cosine     {cpc:+.4f}   (->1.0 = collapse; health <0.98)")
    print(f"  effective rank (entropy)  {ent_rank:8.1f}   (of {D}; higher = more dims used)")
    print(f"  participation ratio       {part_ratio:8.1f}")
    print(f"  linear-probe R2(eval)     {r2_eval:+.4f}   (val; how much Stockfish eval is linearly decodable)")
    print(f"  linear-probe R2(outcome)  {r2_out:+.4f}   (val; STM game outcome)")
    print(f"  decisive sign-accuracy    {sign_acc:.3f}   (val; win/loss positions only)")
    print(f"\nSUMMARY step={step} M={M} cpc={cpc:.4f} eff_rank={ent_rank:.1f} "
          f"r2_eval={r2_eval:.4f} r2_out={r2_out:.4f} sign_acc={sign_acc:.3f}")


if __name__ == "__main__":
    main()
