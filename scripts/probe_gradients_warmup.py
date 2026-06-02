"""Follow-up: does gradient reach the world-model body once the zero-init
prediction heads de-zero? Tracks per-group gradient + dynamics action-blindness
(cross-action cosine) over 200 real training steps.

Tests whether the step-0 "body gets zero gradient" finding is a transient
(heads grow → gradient flows) or a persistent trap, and whether consistency
loss is required to train the body during cold start.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork, _min_max_normalize
from src.model.utils import scalar_to_support

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, HF, B, CW = 5, 8, 24, 2.0
game = ChessGame(); A = game.action_space_size


def fresh_net(embed_dim=16, use_consistency=False):
    torch.manual_seed(0)
    return MuZeroNetwork(
        observation_channels=game.num_planes * HF, action_space_size=A,
        hidden_planes=128, num_blocks=8, latent_h=8, latent_w=8,
        input_h=8, input_w=8, fc_hidden=128, value_support_size=2,
        reward_support_size=1, action_embed_dim=embed_dim,
        use_consistency_loss=use_consistency, proj_hid=1024, proj_out=1024,
        pred_hid=512, pred_out=1024, use_scalar_transform=False,
        value_target_scale=2.0, value_head_type="wdl", draw_score=-0.05,
    ).to(DEV)


def rollout(n=24):
    s = game.reset(); fr, ac = [], []
    for _ in range(n):
        if s.done: break
        fr.append(game.to_tensor(s)); a = int(np.random.choice(game.legal_actions(s)))
        ac.append(a); s, _, _ = game.step(s, a)
    fr.append(game.to_tensor(s)); return fr, ac


def stack(fr, idx):
    return torch.cat([fr[idx - t] if 0 <= idx - t < len(fr) else torch.zeros_like(fr[0])
                      for t in range(HF)], dim=0)


def build_batch():
    obs, act, tobs = [], [], []
    while len(obs) < B:
        fr, ac = rollout()
        for p in range(len(ac) - K):
            if len(obs) >= B: break
            obs.append(stack(fr, p)); act.append(ac[p:p + K])
            tobs.append([stack(fr, p + 1 + k) for k in range(K)])
    return (torch.stack(obs).to(DEV), torch.tensor(act, dtype=torch.long, device=DEV),
            torch.stack([torch.stack(t) for t in tobs]).to(DEV))


obs, actions, tgt_obs = build_batch()
# probe: fixed root + 16 distinct actions → cross-action cosine of dynamics output
probe_h_obs = obs[:1]
probe_acts = torch.randint(0, A, (16,), device=DEV)
draw_tgt = torch.tensor(np.tile([0.02, 0.96, 0.02], (B, 1)).astype(np.float32), device=DEV)
dec = np.zeros((B, 3), np.float32); dec[0::2] = [1, 0, 0]; dec[1::2] = [0, 0, 1]
dec_tgt = torch.tensor(dec, device=DEV)
unif = torch.full((B, A), 1.0 / A, device=DEV)

GROUPS = {"repr": "representation.", "dyn.action_embed": "dynamics.action_embedding.",
          "dyn.body": ("dynamics.conv_in.", "dynamics.bn_in.", "dynamics.blocks."),
          "pred.value": "prediction.value_head."}


def gnorm(net, used):
    out = {}
    for g, pref in GROUPS.items():
        prefs = pref if isinstance(pref, tuple) else (pref,)
        t = sum(p.grad.float().norm().item() ** 2 for n, p in net.named_parameters()
                if any(n.startswith(x) for x in prefs) and p.grad is not None)
        out[g] = t ** 0.5
    emb = net.dynamics.action_embedding.weight.grad
    out["dyn.action_embed"] = emb[used].norm().item() if emb is not None else 0.0
    return out


@torch.no_grad()
def cross_action_cos(net):
    net.eval()
    outs = [net.dynamics(_min_max_normalize(net.representation(probe_h_obs)), a.view(1))[0].flatten()
            for a in probe_acts]
    o = F.normalize(torch.stack(outs), dim=-1); s = o @ o.T
    net.train()
    return s[~torch.eye(16, dtype=bool, device=DEV)].mean().item()


def step_once(net, opt, vt, pt, use_consistency):
    net.train(); opt.zero_grad()
    hidden, pl, vl = net.initial_inference_logits(obs)
    ploss = -(pt * F.log_softmax(pl, 1)).sum(1)
    vloss = -(vt * F.log_softmax(vl, 1)).sum(1)
    rloss = torch.zeros(B, device=DEV); closs = torch.zeros(B, device=DEV)
    for k in range(K):
        hidden, rl, pl, vl = net.recurrent_inference_logits(hidden, actions[:, k])
        hidden.register_hook(lambda g: g * 0.5)
        ploss = ploss + (-(pt * F.log_softmax(pl, 1)).sum(1))
        vloss = vloss + (-(vt * F.log_softmax(vl, 1)).sum(1))
        td = scalar_to_support(torch.zeros(B, device=DEV), 1).to(rl.device)
        rloss = rloss + (-(td * F.log_softmax(rl, 1)).sum(1))
        if use_consistency:
            dp = F.normalize(net.project(hidden, with_grad=True), dim=-1)
            with torch.no_grad():
                tp = F.normalize(net.project(net.representation(tgt_obs[:, k]), with_grad=False), dim=-1)
            closs = closs + (-(dp * tp).sum(-1))
    loss = ((ploss + vloss + rloss + CW * closs) / (K + 1)).mean()
    loss.backward()
    g = gnorm(net, torch.unique(actions))
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return g


def trial(label, vt, pt, use_consistency, embed=16, steps=200):
    net = fresh_net(embed, use_consistency)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    print(f"\n=== {label} ===")
    print(f"  {'step':>4} | {'repr':>8} {'dyn.body':>9} {'act_emb':>9} {'pred.val':>9} | cross-act-cos")
    for s in range(steps + 1):
        g = step_once(net, opt, vt, pt, use_consistency)
        if s in (0, 1, 5, 20, 50, 100, 200):
            print(f"  {s:>4} | {g['repr']:>8.4f} {g['dyn.body']:>9.4f} "
                  f"{g['dyn.action_embed']:>9.5f} {g['pred.value']:>9.4f} | {cross_action_cos(net):.4f}")


trial("consistency OFF, DRAW basin, embed=16", draw_tgt, unif, False)
trial("consistency OFF, DECISIVE, embed=16", dec_tgt, unif, False)
trial("consistency ON,  DRAW basin, embed=16", draw_tgt, unif, True)
trial("consistency OFF, DRAW basin, embed=256", draw_tgt, unif, False, embed=256)
trial("consistency ON,  DRAW basin, embed=256", draw_tgt, unif, True, embed=256)
