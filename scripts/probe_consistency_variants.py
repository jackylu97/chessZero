"""Which consistency-target variant actually breaks dynamics action-blindness?

Trains 200 real steps under each variant on draw-basin targets and reports the
final cross-action cosine (1.0 = action-blind). Decides whether single-frame
SimSiam is enough or we need a contrastive (negatives) objective.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork, _min_max_normalize
from src.model.utils import scalar_to_support

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, HF, B, CW, NP = 5, 8, 24, 2.0, ChessGame().num_planes
game = ChessGame(); A = game.action_space_size


def fresh_net():
    torch.manual_seed(0)
    return MuZeroNetwork(observation_channels=NP*HF, action_space_size=A, hidden_planes=128,
        num_blocks=8, latent_h=8, latent_w=8, input_h=8, input_w=8, fc_hidden=128,
        value_support_size=2, reward_support_size=1, action_embed_dim=16, use_consistency_loss=True,
        proj_hid=1024, proj_out=1024, pred_hid=512, pred_out=1024, use_scalar_transform=False,
        value_target_scale=2.0, value_head_type="wdl", draw_score=-0.05).to(DEV)


def rollout(n=24):
    s = game.reset(); fr, ac = [], []
    for _ in range(n):
        if s.done: break
        fr.append(game.to_tensor(s)); a = int(np.random.choice(game.legal_actions(s)))
        ac.append(a); s, _, _ = game.step(s, a)
    fr.append(game.to_tensor(s)); return fr, ac


def stack(fr, idx):
    return torch.cat([fr[idx-t] if 0<=idx-t<len(fr) else torch.zeros_like(fr[0]) for t in range(HF)], 0)


def build_batch():
    obs, act, tobs = [], [], []
    while len(obs) < B:
        fr, ac = rollout()
        for p in range(len(ac)-K):
            if len(obs) >= B: break
            obs.append(stack(fr, p)); act.append(ac[p:p+K])
            tobs.append([stack(fr, p+1+k) for k in range(K)])
    return (torch.stack(obs).to(DEV), torch.tensor(act, dtype=torch.long, device=DEV),
            torch.stack([torch.stack(t) for t in tobs]).to(DEV))


obs, actions, tgt_obs = build_batch()
probe_h = obs[:1]; probe_acts = torch.randint(0, A, (16,), device=DEV)
draw_tgt = torch.tensor(np.tile([0.02,0.96,0.02],(B,1)).astype(np.float32), device=DEV)
unif = torch.full((B,A), 1.0/A, device=DEV)


def make_target_obs(stk, mode):
    if mode == "8frame":
        return stk
    if mode == "sf_zero":
        m = torch.zeros_like(stk); m[:, :NP] = stk[:, :NP]; return m
    if mode == "sf_rep":
        return stk[:, :NP].repeat(1, HF, 1, 1)
    raise ValueError(mode)


@torch.no_grad()
def cross_action_cos(net):
    net.eval()
    o = F.normalize(torch.stack([net.dynamics(_min_max_normalize(net.representation(probe_h)), a.view(1))[0].flatten()
                                 for a in probe_acts]), dim=-1)
    net.train(); s = o@o.T
    return s[~torch.eye(16, dtype=bool, device=DEV)].mean().item()


def train_variant(mode, contrastive=False, steps=200):
    net = fresh_net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for step in range(steps+1):
        net.train(); opt.zero_grad()
        hidden, pl, vl = net.initial_inference_logits(obs)
        ploss = -(unif*F.log_softmax(pl,1)).sum(1); vloss = -(draw_tgt*F.log_softmax(vl,1)).sum(1)
        rloss = torch.zeros(B,device=DEV); closs = torch.zeros(B,device=DEV)
        for k in range(K):
            hidden, rl, pl, vl = net.recurrent_inference_logits(hidden, actions[:,k])
            hidden.register_hook(lambda g: g*0.5)
            ploss = ploss + (-(unif*F.log_softmax(pl,1)).sum(1))
            vloss = vloss + (-(draw_tgt*F.log_softmax(vl,1)).sum(1))
            td = scalar_to_support(torch.zeros(B,device=DEV),1).to(rl.device)
            rloss = rloss + (-(td*F.log_softmax(rl,1)).sum(1))
            if mode != "off":
                p = F.normalize(net.project(hidden, with_grad=True), dim=-1)
                with torch.no_grad():
                    z = F.normalize(net.project(net.representation(make_target_obs(tgt_obs[:,k], mode)), with_grad=False), dim=-1)
                if contrastive:
                    # InfoNCE: align p_i with z_i, push from z_j (in-batch negatives)
                    logits = (p @ z.T) / 0.1                      # (B,B)
                    closs = closs + F.cross_entropy(logits, torch.arange(B, device=DEV), reduction="none")
                else:
                    closs = closs + (-(p*z).sum(-1))
        loss = ((ploss + vloss + rloss + CW*closs)/(K+1)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return cross_action_cos(net)


print("Final cross-action cos after 200 steps (draw basin). 1.0 = action-blind, lower = better.\n")
for label, mode, contr in [
    ("consistency OFF",                "off",     False),
    ("SimSiam 8-frame (current)",      "8frame",  False),
    ("SimSiam single-frame zero-pad",  "sf_zero", False),
    ("SimSiam single-frame repeat-8x", "sf_rep",  False),
    ("Contrastive(InfoNCE) 8-frame",   "8frame",  True),
    ("Contrastive(InfoNCE) sf-zero",   "sf_zero", True),
    ("Contrastive(InfoNCE) sf-repeat", "sf_rep",  True),
]:
    c = train_variant(mode, contrastive=contr)
    print(f"  {label:32s} -> cross-action cos = {c:.4f}")
