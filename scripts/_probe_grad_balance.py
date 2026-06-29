"""Per-loss-term gradient balance into the VALUE head.

Runs one real batch, then for each loss term separately backprops and measures
the L2 grad norm landing on the value head vs the shared representation/dynamics
backbone. Reveals whether the value head's gradient is being swamped by
consistency / policy / inverse losses (a mechanism that would starve value-head
within-position resolution).
"""
import sys, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/chessZero")
np.random.seed(0); torch.manual_seed(0)

from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer
from scripts.eval_checkpoint_health import build_network

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.pt"
cfg = get_config("chess_small"); cfg.device = "cpu"
game = ChessGame()
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
net = build_network(ckpt, game, cfg, "cpu")
net.train()
step = ckpt.get("step", 0)

rb = ReplayBuffer(cfg.replay_buffer_size); rb.load(CKPT.replace(".pt",".buf"), game=game)
beta = cfg.per_beta_init + (1.0-cfg.per_beta_init)*(int(step)/max(1,cfg.training_steps))
batch, gidx, isw = rb.sample_batch(
    cfg.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount, game.action_space_size,
    alpha=cfg.per_alpha, beta=beta, value_head_type=cfg.value_head_type,
    history_frames=cfg.history_frames, eval_to_wdl_alpha=cfg.eval_to_wdl_alpha,
    eval_to_wdl_beta=cfg.eval_to_wdl_beta, decisive_sample_frac=cfg.decisive_sample_frac,
    q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
    selfplay_q_ratio=cfg.selfplay_q_ratio, repetition_penalty=cfg.repetition_penalty,
    repetition_penalty_decay=cfg.repetition_penalty_decay)

obs = batch["observations"]; actions = batch["actions"]
tv = batch["target_values"]; tp = batch["target_policies"]; tr_ = batch["target_rewards"]
mask = batch["target_obs_mask"]; tobs = batch["target_observations"]
K = cfg.num_unroll_steps; outer = 1.0/(K+1)

def value_head_params():
    return [p for n,p in net.named_parameters() if n.startswith("prediction.value_head.")]
def repr_params():
    return [p for n,p in net.named_parameters() if n.startswith("representation.")]

def gradnorm(loss, params):
    net.zero_grad()
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    acc = 0.0
    for x in g:
        if x is not None:
            acc += float((x.float()**2).sum())
    return acc ** 0.5

def neg_cos(p,z):
    p=F.normalize(p,dim=-1); z=F.normalize(z,dim=-1); return -(p*z).sum(-1)

# forward full unroll, collecting per-term losses (NO autocast — cpu)
hidden, plog, vlog = net.initial_inference_logits(obs)
vh = value_head_params(); rp = repr_params()

# root value loss
val_loss = (-(tv[:,0]*F.log_softmax(vlog,dim=1)).sum(1)).mean()
pol_loss = (-(tp[:,0]*F.log_softmax(plog,dim=1)).sum(1)).mean()
cons_loss = torch.zeros(())
inv_loss = torch.zeros(())
hidden_k = hidden
for k in range(K):
    hin = hidden_k
    hidden_k, rlog, plog_k, vlog_k = net.recurrent_inference_logits(hidden_k, actions[:,k])
    mk = mask[:,k+1]
    val_loss = val_loss + outer*( (-(tv[:,k+1]*F.log_softmax(vlog_k,dim=1)).sum(1))*mk ).mean()
    pol_loss = pol_loss + outer*( (-(tp[:,k+1]*F.log_softmax(plog_k,dim=1)).sum(1))*mk ).mean()
    dp = net.project(hidden_k, with_grad=True)
    with torch.no_grad():
        th = net.representation(tobs[:,k+1]); tpj = net.project(th, with_grad=False)
    cons_loss = cons_loss + outer*( neg_cos(dp,tpj)*mk ).mean()
    ilog = net.predict_inverse_action(hin, hidden_k)
    inv_loss = inv_loss + outer*( F.cross_entropy(ilog, actions[:,k], reduction="none")*mk ).mean()

vw = cfg.value_loss_weight_selfplay  # 1.0 (all self-play here)
cw = cfg.consistency_loss_weight     # 2.0
print(f"=== {CKPT} step={step} ===")
print("Per-term scalar loss (×outer already in unroll terms; root full weight):")
print(f"  value  (raw)        = {val_loss.item():.4f}  applied_weight={vw}")
print(f"  policy (raw)        = {pol_loss.item():.4f}  applied_weight=1.0")
print(f"  consistency (raw)   = {cons_loss.item():.4f}  applied_weight={cw}  -> weighted={cw*cons_loss.item():.4f}")
print(f"  inverse (raw)       = {inv_loss.item():.4f}  applied_weight={cfg.inverse_dynamics_loss_weight}")

print("\nGrad L2 into VALUE HEAD from each term (×its applied weight):")
print(f"  value*{vw}     -> value_head: {gradnorm(vw*val_loss, vh):.4e}")
print(f"  policy        -> value_head: {gradnorm(pol_loss, vh):.4e}  (should be ~0; value head not in policy graph)")
print(f"  consist*{cw}   -> value_head: {gradnorm(cw*cons_loss, vh):.4e}  (should be ~0)")

print("\nGrad L2 into REPRESENTATION (shared backbone) from each term (×weight):")
gv = gradnorm(vw*val_loss, rp)
gp = gradnorm(pol_loss, rp)
gc = gradnorm(cw*cons_loss, rp)
gi = gradnorm(cfg.inverse_dynamics_loss_weight*inv_loss, rp)
print(f"  value*{vw}     -> repr: {gv:.4e}")
print(f"  policy        -> repr: {gp:.4e}")
print(f"  consist*{cw}   -> repr: {gc:.4e}")
print(f"  inverse       -> repr: {gi:.4e}")
tot = gv+gp+gc+gi
print(f"  => value's SHARE of backbone gradient = {gv/tot*100:.1f}%  (policy {gp/tot*100:.0f}%, consist {gc/tot*100:.0f}%, inverse {gi/tot*100:.0f}%)")
