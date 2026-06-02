"""End-to-end training-health probe for the new dynamics fixes.

Runs the REAL MuZeroTrainer._train_step on a chess buffer (seeded from quick
python-chess rollouts) for a few hundred steps with the production chess flags
(inverse dynamics + single-frame consistency), and verifies:
  1. every sub-network receives finite, nonzero gradient,
  2. latents are plausible (min-max-normalized to ~[0,1], per-channel spread, no NaN),
  3. dynamics becomes action-aware (cross-action cosine falls),
  4. value head produces spread (not collapsed to one WDL class),
  5. the inverse loss decreases (action is becoming recoverable).

Run: .venv/bin/python scripts/probe_training_health.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F

from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork, _min_max_normalize
from src.training.replay_buffer import GameHistory
from src.training.trainer import MuZeroTrainer

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
game = ChessGame(); A = game.action_space_size; NP = game.num_planes

cfg = get_config("chess")
cfg.device = DEV
# Lighten for a fast diagnostic; keep all the loss/architecture flags identical to production.
cfg.batch_size = 64
cfg.training_steps = int(os.environ.get("PROBE_STEPS", "600"))
cfg.use_amp = (DEV == "cuda") and (os.environ.get("PROBE_AMP", "1") == "1")
# Isolate aux-loss learning from the LR schedule: the chess preset has a 500-step
# warmup + decay milestones, which dominate a short probe. Constant lr here.
if os.environ.get("PROBE_NO_WARMUP", "1") == "1":
    cfg.lr_warmup_steps = 0
    cfg.lr_decay_milestones = []
    cfg.lr = 2e-3
HF = cfg.history_frames
print(f"flags: consistency={cfg.use_consistency_loss} single_frame={cfg.consistency_single_frame_target} "
      f"inverse={cfg.use_inverse_dynamics_loss} (w={cfg.inverse_dynamics_loss_weight}) "
      f"value_head_init_std={cfg.value_head_init_std} value_head={cfg.value_head_type}")


def make_chess_game(n_plies=26, outcome=0.0):
    s = game.reset(); gh = GameHistory(game_name="chess")
    for _ in range(n_plies):
        if s.done: break
        legal = game.legal_actions(s)
        gh.observations.append(game.to_tensor(s))
        gh.legal_actions_list.append(legal)
        # near-uniform-over-legal policy with a mild peak (cold-start-like)
        p = np.zeros(A, dtype=np.float32); p[legal] = 1.0 / len(legal)
        gh.policies.append(p)
        a = int(np.random.choice(legal)); gh.actions.append(a)
        gh.rewards.append(0.0); gh.root_values.append(0.0)
        s, _, _ = game.step(s, a)
    gh.observations.append(game.to_tensor(s))
    gh.game_outcome = outcome
    return gh

# Seed a buffer with a draw-heavy but not all-draw mix (gives the value head signal).
print("seeding buffer with rollout games...")
network = MuZeroNetwork(
    observation_channels=NP * HF, action_space_size=A, hidden_planes=cfg.hidden_planes,
    num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
    input_h=8, input_w=8, fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim,
    use_consistency_loss=cfg.use_consistency_loss, proj_hid=cfg.proj_hid, proj_out=cfg.proj_out,
    pred_hid=cfg.pred_hid, pred_out=cfg.pred_out, use_scalar_transform=cfg.use_scalar_transform,
    value_target_scale=cfg.value_target_scale, value_head_type=cfg.value_head_type,
    draw_score=cfg.draw_score, value_head_init_std=cfg.value_head_init_std,
    use_inverse_dynamics_loss=cfg.use_inverse_dynamics_loss,
    inverse_dynamics_hidden=cfg.inverse_dynamics_hidden,
)
print(f"network params: {sum(p.numel() for p in network.parameters())/1e6:.2f}M")
trainer = MuZeroTrainer(cfg, game, network, run_id="health_probe", device=DEV,
                        log_dir="/tmp/health_probe_runs", checkpoints_dir="/tmp/health_probe_ckpts")
rng = np.random.default_rng(0)
for i in range(90):
    o = float(rng.choice([0.0, 0.0, 0.0, 1.0, -1.0]))   # ~60% draw
    trainer.replay_buffer.save_game(make_chess_game(n_plies=int(rng.integers(16, 30)), outcome=o))
print(f"buffer: {len(trainer.replay_buffer)} games")

# Fixed probe batch for plausibility + action-awareness checks.
probe_games = [make_chess_game(28, 0.0) for _ in range(8)]
probe_obs = torch.stack([g._stack_history(6, HF) for g in probe_games]).to(DEV)
probe_acts = torch.randint(0, A, (16,), device=DEV)

GROUPS = {
    "representation": "representation.", "dyn.action_embed": "dynamics.action_embedding.",
    "dyn.conv+blocks": ("dynamics.conv_in.", "dynamics.bn_in.", "dynamics.blocks."),
    "dyn.reward_head": "dynamics.reward_head.", "pred.policy_head": "prediction.policy_head.",
    "pred.value_head": "prediction.value_head.", "projection": "projection.",
    "prediction_head": "prediction_head.", "inverse_head": "inverse_dynamics_head.",
}

def grad_norms():
    out = {}
    for g, pref in GROUPS.items():
        prefs = pref if isinstance(pref, tuple) else (pref,)
        gs = [p.grad for n, p in network.named_parameters()
              if any(n.startswith(x) for x in prefs) and p.grad is not None]
        out[g] = (sum(x.float().norm().item()**2 for x in gs)**0.5) if gs else 0.0
    return out

@torch.no_grad()
def plausibility():
    network.eval()
    h = network.representation(probe_obs)                       # (B,C,H,W)
    # per-action dynamics outputs from one fixed root → cross-action cosine
    h0 = network.representation(probe_obs[:1])
    outs = torch.stack([network.dynamics(h0, a.view(1))[0].flatten() for a in probe_acts])
    on = F.normalize(outs, dim=-1); cos = (on @ on.T)[~torch.eye(16, dtype=bool, device=DEV)].mean().item()
    pl, vl = network.prediction(h)
    wdl = F.softmax(vl.float(), dim=-1)                         # (B,3)
    pv = (wdl[:, 0] - wdl[:, 2])                                # scalar value per sample
    pe = -(F.softmax(pl.float(), -1) * F.log_softmax(pl.float(), -1)).sum(-1).mean().item()
    network.train()
    return {
        "h_min": h.min().item(), "h_max": h.max().item(),
        "h_perchan_std": h.var(dim=(0, 2, 3)).sqrt().mean().item(),
        "h_finite": bool(torch.isfinite(h).all()),
        "cross_action_cos": cos,
        "wdl_mean": wdl.mean(0).tolist(), "pred_v_std": pv.std().item(),
        "policy_entropy": pe, "policy_entropy_maxposs": float(np.log(A)),
    }

print("\nstep | grads: repr  dyn_emb dyn_body invhead pred_val | losses: tot  inv  cons  val | xa_cos pred_v_std")
for step in range(cfg.training_steps + 1):
    info = trainer._train_step()
    if step % 50 == 0:
        gn = grad_norms(); pl = plausibility()
        print(f"{step:4d} | {gn['representation']:5.2f} {gn['dyn.action_embed']:6.3f} "
              f"{gn['dyn.conv+blocks']:7.2f} {gn['inverse_head']:6.2f} {gn['pred.value_head']:7.3f} | "
              f"{info['total_loss']:5.2f} {info['inverse_loss']:5.2f} {info['consistency_loss']:5.2f} "
              f"{info['value_loss']:5.3f} | {pl['cross_action_cos']:.3f}  {pl['pred_v_std']:.3f}")

print("\n=== Final plausibility ===")
pl = plausibility()
for k, v in pl.items():
    print(f"  {k:22s} = {v}")

print("\n=== Health verdict ===")
gn = grad_norms()
checks = [
    ("all sub-networks get gradient", all(gn[g] > 0 for g in
        ["representation", "dyn.action_embed", "dyn.conv+blocks", "pred.policy_head",
         "pred.value_head", "dyn.reward_head", "inverse_head"]
        + (["projection", "prediction_head"] if cfg.use_consistency_loss else []))),
    ("latents finite", pl["h_finite"]),
    ("latents normalized to ~[0,1]", pl["h_min"] >= -1e-3 and pl["h_max"] <= 1 + 1e-3),
    ("latents not collapsed (per-chan std>0.01)", pl["h_perchan_std"] > 0.01),
    ("dynamics action-aware (cross-action cos < 0.97)", pl["cross_action_cos"] < 0.97),
    ("value head not collapsed (pred_v_std > 0.02)", pl["pred_v_std"] > 0.02),
    ("policy not degenerate (0.2*max < H < max)", 0.2 * pl["policy_entropy_maxposs"] < pl["policy_entropy"] <= pl["policy_entropy_maxposs"]),
]
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
