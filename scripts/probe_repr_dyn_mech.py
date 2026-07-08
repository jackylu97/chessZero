#!/usr/bin/env python3
"""Mechanistic degeneracy probe for representation + dynamics, over checkpoints.

For each checkpoint (CPU), sample positions from the matching self-play .buf and:

REPRESENTATION
  - per-dim std, cross-position cosine, effective rank, participation ratio
  - linear-probe R2(outcome) decodability from frozen latent (buffer outcome)

DYNAMICS (the within-position blindness suspect)
  - cross-action cosine: cos(dyn(h,a_i), dyn(h,a_j)) over legal moves at a root
    (1.0 => action-blind = real mechanistic bug)
  - per-position dyn spread (same metric, per-root mean)
  - cos(root, dyn): is dynamics ~= identity?
  - inverse-action recovery accuracy (is action recoverable from (h, h_next)?)
  - consistency: cos(dyn(h,a), repr(true next obs)) — does the world model track reality?
  - VALUE spread across actions from a root via prediction(dyn(h,a)) — the actual
    quantity MCTS needs to rank siblings. This is the direct within-position signal.
  - K-step rollout fixed-point check: does recurrent latent stop moving / collapse to
    a constant as K grows?

Run: .venv/bin/python scripts/probe_repr_dyn_mech.py
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.representation_probe import effective_rank, mean_crosspos_cosine, dual_ridge_r2
from src.model.utils import wdl_to_scalar

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"


def build_network(ckpt, game, cfg, device):
    sd = ckpt["model_state_dict"]
    has_conv_policy = any(".policy_head.mix." in k or ".policy_head.proj." in k for k in sd)
    has_moves_left = any(k.startswith("moves_left_head.") for k in sd)
    has_consistency = any(k.startswith("projection.") for k in sd)
    has_inverse = any(k.startswith("inverse_dynamics_head.") for k in sd)
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim,
        use_consistency_loss=has_consistency, proj_hid=cfg.proj_hid, proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid, pred_out=cfg.pred_out, use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale, value_head_type=cfg.value_head_type,
        draw_score=cfg.draw_score, value_head_init_std=getattr(cfg, "value_head_init_std", 0.0),
        use_inverse_dynamics_loss=has_inverse,
        inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 256),
        policy_head_type="conv" if has_conv_policy else "flat",
        use_moves_left=has_moves_left,
        moves_left_support_size=getattr(cfg, "moves_left_support_size", 10),
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


def load_buf_games(path, game, max_games):
    games = []
    with open(path, "rb") as f:
        first = pickle.load(f)
        n = first["n_records"]
        ver = first["version"]
        from src.training.replay_buffer import GameHistory
        for _ in range(n):
            record, priority = pickle.load(f)
            if ver == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if len(games) >= max_games:
                break
    return games


def sample_positions(games, hf, n_positions, rng):
    """Return list of (game, ply) sampled across games, plus stacked obs."""
    picks = []
    for gh in games:
        n = len(gh.observations)
        if n < 2:
            continue
        k = max(1, min(8, n // 4))
        plies = rng.choice(n - 1, size=min(k, n - 1), replace=False)  # avoid last ply (no action)
        for ply in plies:
            picks.append((gh, int(ply)))
    rng.shuffle(picks)
    picks = picks[:n_positions]
    obs = []
    for gh, ply in picks:
        obs.append(gh._stack_history(ply, hf).numpy())
    return picks, np.asarray(obs, dtype=np.float32)


@torch.no_grad()
def run_checkpoint(path, game, cfg, dev, n_positions, n_games, seed):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    step = ckpt.get("step", "?")
    net = build_network(ckpt, game, cfg, dev)
    hf = cfg.history_frames
    rng = np.random.default_rng(seed)

    bufpath = path.replace(".pt", ".buf")
    games = load_buf_games(bufpath, game, n_games)
    picks, obs = sample_positions(games, hf, n_positions, rng)
    M = len(picks)

    out = {"step": step, "M": M}

    # ---- REPRESENTATION ----
    H = []
    for s in range(0, M, 256):
        x = torch.from_numpy(obs[s:s+256]).to(dev)
        h = net.representation(x).reshape(x.shape[0], -1).float().cpu().numpy()
        H.append(h)
    H = np.concatenate(H, 0)
    D = H.shape[1]
    out["D"] = D
    out["per_dim_std"] = float(H.std(0).mean())
    out["cpc"] = mean_crosspos_cosine(H.astype(np.float64), seed=seed)
    er, pr = effective_rank(H.astype(np.float64))
    out["eff_rank"] = er
    out["part_ratio"] = pr
    # outcome label STM-relative
    outc = []
    for gh, ply in picks:
        stm_white = (ply % 2 == 0)
        outc.append(float(gh.game_outcome) * (1.0 if stm_white else -1.0))
    outc = np.asarray(outc)
    out["wdl_frac"] = ((outc > 0.5).mean(), (np.abs(outc) <= 0.5).mean(), (outc < -0.5).mean())
    r2_out, _ = dual_ridge_r2(H.astype(np.float64), outc.astype(np.float64), seed=seed)
    out["r2_out"] = r2_out

    # ---- DYNAMICS ----
    # Use a subset of roots with their legal actions for action-conditioning probes.
    cross_cos, per_pos_spread, cos_root_dyn = [], [], []
    inv_acc = []
    consistency_cos = []
    dyn_value_std = []   # value spread across sibling actions (the key within-pos signal)
    root_values = []
    n_dyn_roots = min(120, M)
    for i in range(n_dyn_roots):
        gh, ply = picks[i]
        legal = gh.legal_actions_list[ply] if ply < len(gh.legal_actions_list) else []
        if len(legal) < 2:
            continue
        x = torch.from_numpy(obs[i:i+1]).to(dev)
        h = net.representation(x)
        # root value
        _, vl = net.prediction(h)
        root_values.append(wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item())
        probe = legal[:24]
        acts = torch.tensor(probe, dtype=torch.long, device=dev)
        hn = h.expand(len(probe), *h.shape[1:]).contiguous()
        dyn, _ = net.dynamics(hn, acts)
        dv = F.normalize(dyn.reshape(len(probe), -1), dim=-1)
        cc = (dv @ dv.T)[~torch.eye(len(probe), dtype=bool)].mean().item()
        cross_cos.append(cc)
        per_pos_spread.append(cc)
        # cos(root, dyn): identity test
        hflat = F.normalize(h.reshape(1, -1), dim=-1)
        cos_root_dyn.append((hflat @ dv.T).mean().item())
        # value spread across actions
        _, vla = net.prediction(dyn)
        avs = wdl_to_scalar(vla.float(), draw_score=cfg.draw_score)
        dyn_value_std.append(avs.std().item())
        # inverse-action recovery
        if net.inverse_dynamics_head is not None:
            logits = net.predict_inverse_action(hn, dyn)
            inv_acc.append((logits.argmax(-1) == acts).float().mean().item())
        # consistency: dyn(h, a) vs repr(true next obs). Use the ACTUAL played action.
        a_played = gh.actions[ply]
        # build true next obs stack
        if ply + 1 < len(gh.observations):
            nobs = torch.from_numpy(gh._stack_history(ply + 1, hf).numpy()).unsqueeze(0).to(dev)
            h_next_true = net.representation(nobs)
            a_t = torch.tensor([a_played], dtype=torch.long, device=dev)
            h_dyn, _ = net.dynamics(h, a_t)
            consistency_cos.append(F.cosine_similarity(h_dyn.flatten(), h_next_true.flatten(), dim=0).item())

    out["cross_action_cos"] = float(np.mean(cross_cos)) if cross_cos else float("nan")
    out["per_pos_dyn_spread"] = float(np.mean(per_pos_spread)) if per_pos_spread else float("nan")
    out["cos_root_dyn"] = float(np.mean(cos_root_dyn)) if cos_root_dyn else float("nan")
    out["inv_recovery"] = float(np.mean(inv_acc)) if inv_acc else float("nan")
    out["consistency_cos"] = float(np.mean(consistency_cos)) if consistency_cos else float("nan")
    out["dyn_value_std_per_root"] = float(np.mean(dyn_value_std)) if dyn_value_std else float("nan")
    out["root_value_std"] = float(np.std(root_values)) if root_values else float("nan")
    out["root_value_mean"] = float(np.mean(root_values)) if root_values else float("nan")

    # ---- K-STEP ROLLOUT FIXED-POINT CHECK ----
    # Roll the dynamics forward K steps along the ACTUAL game actions from a few roots
    # and measure (a) cos(h_k, h_{k-1}) (is it stalling?) and (b) cross-position cos of
    # h_k across distinct roots (is it converging to a shared fixed point?).
    K = 5
    rollout_latents = [[] for _ in range(K + 1)]
    step_to_step_cos = [[] for _ in range(K)]
    n_roll = min(40, M)
    for i in range(n_roll):
        gh, ply = picks[i]
        if ply + K >= len(gh.observations):
            continue
        x = torch.from_numpy(obs[i:i+1]).to(dev)
        h = net.representation(x)
        rollout_latents[0].append(h.reshape(-1).cpu().numpy())
        prev = h
        for k in range(K):
            a = gh.actions[ply + k]
            a_t = torch.tensor([a], dtype=torch.long, device=dev)
            h, _ = net.dynamics(h, a_t)
            rollout_latents[k + 1].append(h.reshape(-1).cpu().numpy())
            step_to_step_cos[k].append(
                F.cosine_similarity(h.flatten(), prev.flatten(), dim=0).item())
            prev = h
    out["rollout_crosspos_cos"] = []
    out["rollout_step_cos"] = []
    for k in range(K + 1):
        if len(rollout_latents[k]) >= 2:
            Hk = np.asarray(rollout_latents[k], dtype=np.float64)
            out["rollout_crosspos_cos"].append(round(mean_crosspos_cosine(Hk, seed=seed), 4))
        else:
            out["rollout_crosspos_cos"].append(float("nan"))
    for k in range(K):
        out["rollout_step_cos"].append(
            round(float(np.mean(step_to_step_cos[k])), 4) if step_to_step_cos[k] else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-positions", type=int, default=1500)
    ap.add_argument("--n-games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default="1000,6000,30000")
    args = ap.parse_args()

    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = args.device
    steps = [int(s) for s in args.steps.split(",")]
    results = []
    for st in steps:
        path = os.path.join(CKDIR, f"checkpoint_{st}.pt")
        r = run_checkpoint(path, game, cfg, args.device, args.n_positions, args.n_games, args.seed)
        results.append(r)
        print(f"\n{'='*72}\nstep {r['step']}  (M={r['M']}, D={r['D']})")
        print(f"  REPR per_dim_std        {r['per_dim_std']:.4f}")
        print(f"  REPR cross-pos cosine   {r['cpc']:+.4f}   (->1.0 collapse)")
        print(f"  REPR eff_rank (entropy) {r['eff_rank']:.1f} / {r['D']}   part_ratio {r['part_ratio']:.1f}")
        print(f"  REPR r2(outcome)        {r['r2_out']:+.4f}   WDL frac {r['wdl_frac'][0]:.2f}/{r['wdl_frac'][1]:.2f}/{r['wdl_frac'][2]:.2f}")
        print(f"  DYN cross-action cos    {r['cross_action_cos']:+.4f}   (1.0=action-blind)")
        print(f"  DYN cos(root,dyn)       {r['cos_root_dyn']:+.4f}   (1.0=identity)")
        print(f"  DYN inverse-recovery    {r['inv_recovery']:.3f}")
        print(f"  DYN consistency cos     {r['consistency_cos']:+.4f}   (dyn vs repr_next)")
        print(f"  DYN value-std/root      {r['dyn_value_std_per_root']:.4f}   (sibling-value spread = within-pos signal)")
        print(f"  root V mean/std         {r['root_value_mean']:+.4f} / {r['root_value_std']:.4f}")
        print(f"  rollout crosspos cos    {r['rollout_crosspos_cos']}   (k=0..5; ->1 = shared fixed point)")
        print(f"  rollout step->step cos  {r['rollout_step_cos']}   (k=1..5; ->1 = stalling)")

    print(f"\n{'='*72}\nTREND")
    keys = [("per_dim_std", ".4f"), ("cpc", "+.4f"), ("eff_rank", ".1f"), ("r2_out", "+.4f"),
            ("cross_action_cos", "+.4f"), ("cos_root_dyn", "+.4f"), ("inv_recovery", ".3f"),
            ("consistency_cos", "+.4f"), ("dyn_value_std_per_root", ".4f"),
            ("root_value_std", ".4f")]
    hdr = "  " + f"{'metric':24s}" + "".join(f"{r['step']:>12}" for r in results)
    print(hdr)
    for k, fmt in keys:
        row = "  " + f"{k:24s}" + "".join(f"{format(r[k], fmt):>12}" for r in results)
        print(row)


if __name__ == "__main__":
    main()
