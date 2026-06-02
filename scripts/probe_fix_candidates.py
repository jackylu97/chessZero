"""Decide the real fix for dynamics action-blindness.

Tests two hypotheses the variant-probe raised:
  (a) memorization artifact — with FRESH diverse data each step (no 24-position
      overfit), does plain single-frame consistency start teaching action-awareness?
  (b) consistency insufficient — does an INVERSE-DYNAMICS aux loss (predict a_k
      from h_k,h_{k+1}; ICM/Pathak) directly force action-encoding?

Metric: cross-action cosine on a held-out fixed position (1.0 = action-blind).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork, _min_max_normalize
from src.model.utils import scalar_to_support

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, HF, B, CW, NP = 5, 8, 24, 2.0, ChessGame().num_planes
game = ChessGame(); A = game.action_space_size
C, Hh, Ww = 128, 8, 8


def fresh_net():
    torch.manual_seed(0)
    return MuZeroNetwork(observation_channels=NP*HF, action_space_size=A, hidden_planes=C,
        num_blocks=8, latent_h=Hh, latent_w=Ww, input_h=8, input_w=8, fc_hidden=128,
        value_support_size=2, reward_support_size=1, action_embed_dim=16, use_consistency_loss=True,
        proj_hid=1024, proj_out=1024, pred_hid=512, pred_out=1024, use_scalar_transform=False,
        value_target_scale=2.0, value_head_type="wdl", draw_score=-0.05).to(DEV)


def inverse_head():
    return nn.Sequential(nn.Linear(2*C*Hh*Ww, 256), nn.ReLU(), nn.Linear(256, A)).to(DEV)


def rollout(n=26):
    s = game.reset(); fr, ac = [], []
    for _ in range(n):
        if s.done: break
        fr.append(game.to_tensor(s)); a = int(np.random.choice(game.legal_actions(s)))
        ac.append(a); s, _, _ = game.step(s, a)
    fr.append(game.to_tensor(s)); return fr, ac


def stack(fr, idx):
    return torch.cat([fr[idx-t] if 0<=idx-t<len(fr) else torch.zeros_like(fr[0]) for t in range(HF)], 0)


# Build a LARGE pool of diverse positions (no per-step memorization possible).
pool_obs, pool_act, pool_tobs = [], [], []
while len(pool_obs) < 600:
    fr, ac = rollout()
    for p in range(len(ac)-K):
        pool_obs.append(stack(fr, p)); pool_act.append(ac[p:p+K])
        pool_tobs.append([stack(fr, p+1+k) for k in range(K)])
pool_obs = torch.stack(pool_obs); pool_act = torch.tensor(pool_act, dtype=torch.long)
pool_tobs = torch.stack([torch.stack(t) for t in pool_tobs])
POOL = len(pool_obs)
print(f"pool size = {POOL} positions")
probe_h = pool_obs[:1].to(DEV); probe_acts = torch.randint(0, A, (16,), device=DEV)
draw_tgt = torch.tensor(np.tile([0.02,0.96,0.02],(B,1)).astype(np.float32), device=DEV)
unif = torch.full((B,A), 1.0/A, device=DEV)


def sample(fixed=False):
    idx = torch.arange(B) if fixed else torch.randint(0, POOL, (B,))
    return (pool_obs[idx].to(DEV), pool_act[idx].to(DEV), pool_tobs[idx].to(DEV))


def sf_zero(stk):
    m = torch.zeros_like(stk); m[:, :NP] = stk[:, :NP]; return m


@torch.no_grad()
def cross_action_cos(net):
    net.eval()
    o = F.normalize(torch.stack([net.dynamics(_min_max_normalize(net.representation(probe_h)), a.view(1))[0].flatten()
                                 for a in probe_acts]), dim=-1)
    net.train(); s = o@o.T
    return s[~torch.eye(16, dtype=bool, device=DEV)].mean().item()


def train(mode, steps=300, fixed=False):
    net = fresh_net(); inv = inverse_head()
    params = list(net.parameters()) + (list(inv.parameters()) if "inv" in mode else [])
    opt = torch.optim.Adam(params, lr=2e-3, weight_decay=1e-4)
    traj = []
    for step in range(steps+1):
        obs, actions, tgt_obs = sample(fixed)
        net.train(); opt.zero_grad()
        hidden, pl, vl = net.initial_inference_logits(obs)
        ploss=-(unif*F.log_softmax(pl,1)).sum(1); vloss=-(draw_tgt*F.log_softmax(vl,1)).sum(1)
        rloss=torch.zeros(B,device=DEV); closs=torch.zeros(B,device=DEV); iloss=torch.zeros(B,device=DEV)
        prev = hidden
        for k in range(K):
            hidden, rl, pl, vl = net.recurrent_inference_logits(hidden, actions[:,k])
            hidden.register_hook(lambda g: g*0.5)
            ploss=ploss+(-(unif*F.log_softmax(pl,1)).sum(1)); vloss=vloss+(-(draw_tgt*F.log_softmax(vl,1)).sum(1))
            td=scalar_to_support(torch.zeros(B,device=DEV),1).to(rl.device); rloss=rloss+(-(td*F.log_softmax(rl,1)).sum(1))
            if "cons" in mode:
                p=F.normalize(net.project(hidden,with_grad=True),dim=-1)
                with torch.no_grad():
                    z=F.normalize(net.project(net.representation(sf_zero(tgt_obs[:,k])),with_grad=False),dim=-1)
                closs=closs+(-(p*z).sum(-1))
            if "inv" in mode:
                # predict a_k from (h_k=prev, h_{k+1}=hidden); forces hidden to encode a_k
                feats = torch.cat([prev.reshape(B,-1), hidden.reshape(B,-1)], dim=1)
                iloss = iloss + F.cross_entropy(inv(feats), actions[:,k], reduction="none")
            prev = hidden
        loss=((ploss+vloss+rloss+CW*closs+iloss)/(K+1)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if step in (0,20,50,100,200,300): traj.append((step, cross_action_cos(net)))
    return traj


print("cross-action cos (1.0=action-blind). step:cos\n")
for label, mode, fixed in [
    ("single-frame cons, FIXED 24 (control)",   "cons",      True),
    ("single-frame cons, FRESH diverse data",   "cons",      False),
    ("inverse-dynamics only, fresh",            "inv",       False),
    ("consistency + inverse-dynamics, fresh",   "cons+inv",  False),
]:
    traj = train(mode, fixed=fixed)
    print(f"  {label:40s} " + "  ".join(f"{s}:{c:.3f}" for s,c in traj))
