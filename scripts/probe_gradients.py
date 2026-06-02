"""Gradient-flow probe for the MuZero dynamics/representation networks.

Replicates trainer._train_step's exact loss math (WDL value head, unroll loop,
the 0.5 hidden-state hook, the 1/(K+1) outer scale) on real chess positions,
and instruments per-parameter-group gradient norms — with particular focus on
how much gradient reaches dynamics.action_embedding relative to everything else,
and how the per-unroll-step gradient decays under the 0.5 hook.

Run: .venv/bin/python scripts/probe_gradients.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import scalar_to_support

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"
K = 5                 # num_unroll_steps (chess config)
HF = 8                # history_frames
B = 32                # batch
CW = 2.0              # consistency_loss_weight

game = ChessGame()
A = game.action_space_size


def fresh_net(embed_dim=16, use_consistency=False):
    torch.manual_seed(0)
    return MuZeroNetwork(
        observation_channels=game.num_planes * HF,
        action_space_size=A,
        hidden_planes=128, num_blocks=8, latent_h=8, latent_w=8,
        input_h=8, input_w=8, fc_hidden=128,
        value_support_size=2, reward_support_size=1,
        action_embed_dim=embed_dim,
        use_consistency_loss=use_consistency,
        proj_hid=1024, proj_out=1024, pred_hid=512, pred_out=1024,
        use_scalar_transform=False, value_target_scale=2.0,
        value_head_type="wdl", draw_score=-0.05,
    ).to(DEV)


def rollout(n_plies=24):
    """Play random moves; return per-ply (single_frame_obs, action, legal)."""
    s = game.reset()
    frames, acts, legals = [], [], []
    for _ in range(n_plies):
        if s.done:
            break
        legal = game.legal_actions(s)
        frames.append(game.to_tensor(s))
        a = int(np.random.choice(legal))
        legals.append(legal)
        acts.append(a)
        s, _, _ = game.step(s, a)
    frames.append(game.to_tensor(s))
    return frames, acts, legals


def stack(frames, idx):
    """8-frame history stack ending at idx, newest-first (matches _stack_history)."""
    out = []
    for t in range(HF):
        i = idx - t
        out.append(frames[i] if 0 <= i < len(frames) else torch.zeros_like(frames[0]))
    return torch.cat(out, dim=0)


def build_batch():
    """B real (root, K-action, K+1 target-obs) samples from random playouts."""
    obs, actions, tgt_obs, tgt_mask, legal_at = [], [], [], [], []
    while len(obs) < B:
        frames, acts, legals = rollout()
        # roots must leave room for K unroll steps with real obs
        for p in range(len(acts) - K):
            if len(obs) >= B:
                break
            obs.append(stack(frames, p))
            actions.append(acts[p:p + K])
            tgt_obs.append([stack(frames, p + 1 + k) for k in range(K)])
            tgt_mask.append([1.0] * K)
            legal_at.append(legals[p])
    obs = torch.stack(obs).to(DEV)                                  # (B, 152,8,8)
    actions = torch.tensor(actions, dtype=torch.long, device=DEV)   # (B, K)
    tgt_obs = torch.stack([torch.stack(t) for t in tgt_obs]).to(DEV)  # (B,K,152,8,8)
    tgt_mask = torch.tensor(tgt_mask, device=DEV)                   # (B, K)
    return obs, actions, tgt_obs, tgt_mask, legal_at


def diffuse_policy(legal_at):
    """Near-uniform-over-legal policy targets (cold-start-like)."""
    P = np.zeros((B, A), dtype=np.float32)
    for i, legal in enumerate(legal_at):
        P[i, legal] = 1.0 / len(legal)
    return torch.tensor(P, device=DEV)


GROUPS = {
    "representation": "representation.",
    "dyn.action_embed": "dynamics.action_embedding.",
    "dyn.conv_in": "dynamics.conv_in.",
    "dyn.bn_in": "dynamics.bn_in.",
    "dyn.blocks": "dynamics.blocks.",
    "dyn.reward_head": "dynamics.reward_head.",
    "pred.policy_head": "prediction.policy_head.",
    "pred.value_head": "prediction.value_head.",
    "projection": "projection.",
    "prediction_head": "prediction_head.",
}


def group_grad_norms(net):
    out = {}
    for gname, prefix in GROUPS.items():
        tot = 0.0
        for n, p in net.named_parameters():
            if n.startswith(prefix) and p.grad is not None:
                tot += p.grad.detach().float().norm().item() ** 2
        out[gname] = tot ** 0.5
    return out


def run(net, obs, actions, tgt_obs, tgt_mask, value_target, policy_target,
        hook=True, root_heavy=False, use_consistency=False, label=""):
    net.train()
    net.zero_grad()
    unroll_scale = (1.0 / K) if root_heavy else 1.0
    outer_scale = 1.0 if root_heavy else 1.0 / (K + 1)

    hidden, policy_logits, value_logits = net.initial_inference_logits(obs)
    per_k_hidden_grad = {}

    def vloss(vl, vt):  # WDL cross-entropy
        return -(vt * F.log_softmax(vl, dim=1)).sum(dim=1)

    def ploss(pl, pt):
        return -(pt * F.log_softmax(pl, dim=1)).sum(dim=1)

    def rloss(rl, rt):
        td = scalar_to_support(rt, 1).to(rl.device)
        return -(td * F.log_softmax(rl, dim=1)).sum(dim=1)

    policy_loss = ploss(policy_logits, policy_target)
    value_loss = vloss(value_logits, value_target)
    reward_loss = torch.zeros(B, device=DEV)
    consistency_loss = torch.zeros(B, device=DEV)

    for k in range(K):
        hidden, reward_logits, policy_logits, value_logits = \
            net.recurrent_inference_logits(hidden, actions[:, k])

        def mk(kk):
            def h(grad):
                per_k_hidden_grad[kk] = grad.detach().norm().item()
                return grad * 0.5 if hook else grad
            return h
        hidden.register_hook(mk(k))

        m = tgt_mask[:, k]
        policy_loss = policy_loss + unroll_scale * ploss(policy_logits, policy_target) * m
        value_loss = value_loss + unroll_scale * vloss(value_logits, value_target) * m
        reward_loss = reward_loss + unroll_scale * rloss(reward_logits, torch.zeros(B, device=DEV)) * m
        if use_consistency:
            dyn_proj = net.project(hidden, with_grad=True)
            with torch.no_grad():
                th = net.representation(tgt_obs[:, k])
                tp = net.project(th, with_grad=False)
            dp = F.normalize(dyn_proj, dim=-1); tp = F.normalize(tp, dim=-1)
            consistency_loss = consistency_loss + unroll_scale * (-(dp * tp).sum(-1)) * m

    per_sample = outer_scale * (policy_loss + value_loss + reward_loss + CW * consistency_loss)
    total = per_sample.mean()
    total.backward()

    g = group_grad_norms(net)
    # action-embedding grad restricted to rows actually used (meaningful signal)
    emb = net.dynamics.action_embedding.weight.grad
    used = torch.unique(actions)
    emb_used = emb[used].norm().item() if emb is not None else 0.0
    return {
        "label": label, "total": total.item(), "groups": g,
        "per_k_hidden_grad": [per_k_hidden_grad.get(k, 0.0) for k in range(K)],
        "emb_used_grad": emb_used, "n_used_actions": len(used),
        "consistency": (outer_scale * CW * consistency_loss.mean()).item() if use_consistency else 0.0,
    }


def show(r):
    print(f"\n=== {r['label']} ===")
    print(f"  total_loss={r['total']:.4f}" + (f"  consistency_term={r['consistency']:.4f}" if r['consistency'] else ""))
    g = r["groups"]
    order = ["representation", "dyn.action_embed", "dyn.conv_in", "dyn.bn_in",
             "dyn.blocks", "dyn.reward_head", "pred.policy_head", "pred.value_head",
             "projection", "prediction_head"]
    for k in order:
        if g.get(k, 0.0) > 0:
            print(f"    grad[{k:18s}] = {g[k]:.5f}")
    print(f"    action_embed grad (used rows only, {r['n_used_actions']} actions) = {r['emb_used_grad']:.5f}")
    print(f"    per-unroll-step incoming hidden grad (k=0..{K-1}): "
          + ", ".join(f"{x:.4f}" for x in r["per_k_hidden_grad"]))


print("Building batch of real chess positions (8-frame history)...")
obs, actions, tgt_obs, tgt_mask, legal_at = build_batch()
pol = diffuse_policy(legal_at)

# WDL targets
draw_tgt = torch.tensor(np.tile([0.02, 0.96, 0.02], (B, 1)).astype(np.float32), device=DEV)
# decisive: half win / half loss
dec = np.zeros((B, 3), dtype=np.float32)
dec[0::2] = [1.0, 0.0, 0.0]; dec[1::2] = [0.0, 0.0, 1.0]
dec_tgt = torch.tensor(dec, device=DEV)

print("\n################ REGIME A: DRAW BASIN (value≈draw, diffuse policy) ################")
show(run(fresh_net(16), obs, actions, tgt_obs, tgt_mask, draw_tgt, pol, hook=True, label="draw basin, embed=16, hook ON, 1/(K+1)"))

print("\n################ REGIME B: DECISIVE value targets (still unique h per sample) ################")
show(run(fresh_net(16), obs, actions, tgt_obs, tgt_mask, dec_tgt, pol, hook=True, label="decisive, embed=16, hook ON, 1/(K+1)"))

print("\n################ HOOK on/off and root-heavy (draw basin) ################")
show(run(fresh_net(16), obs, actions, tgt_obs, tgt_mask, draw_tgt, pol, hook=False, label="draw basin, embed=16, hook OFF, 1/(K+1)"))
show(run(fresh_net(16), obs, actions, tgt_obs, tgt_mask, draw_tgt, pol, hook=True, root_heavy=True, label="draw basin, embed=16, hook ON, ROOT-HEAVY"))

print("\n################ EMBED DIM sweep (draw basin) ################")
show(run(fresh_net(64), obs, actions, tgt_obs, tgt_mask, draw_tgt, pol, hook=True, label="draw basin, embed=64, hook ON"))
show(run(fresh_net(256), obs, actions, tgt_obs, tgt_mask, draw_tgt, pol, hook=True, label="draw basin, embed=256, hook ON"))

print("\n################ CONSISTENCY LOSS on (single-frame target via repr of next obs) ################")
show(run(fresh_net(16, True), obs, actions, tgt_obs, tgt_mask, draw_tgt, pol, hook=True, use_consistency=True, label="draw basin, embed=16, CONSISTENCY ON"))
