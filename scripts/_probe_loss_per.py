"""Diagnostic: loss math + weighting + PER sampling integrity.

Loads a real chess_small checkpoint + .buf, runs the ACTUAL sample_batch +
_train_step path on CPU, and dumps:
  - PER priority distribution (are priorities degenerate/uniform?)
  - sampled-index / decisive-stratification distribution
  - per-component loss values + their applied weights
  - IS-weight statistics
  - WDL value-loss target shape / normalization checks
"""
import sys, numpy as np, torch
sys.path.insert(0, "/workspace/chessZero")
torch.manual_seed(0); np.random.seed(0)

from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer
from scripts.eval_checkpoint_health import build_network

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.pt"
BUF = CKPT.replace(".pt", ".buf")

cfg = get_config("chess_small"); cfg.device = "cpu"
game = ChessGame()
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
net = build_network(ckpt, game, cfg, "cpu")
step = ckpt.get("step", "?")
print(f"=== checkpoint {CKPT} step={step} ===")

rb = ReplayBuffer(cfg.replay_buffer_size)
rb.load(BUF, game=game)
print(f"buffer: {len(rb)} games  total_games={rb.total_games}")

# ---- 1. PRIORITY DISTRIBUTION ----
pr = np.array(rb._priorities, dtype=np.float64)
print("\n--- PRIORITIES ---")
print(f"n={len(pr)}  min={pr.min():.4g} max={pr.max():.4g} mean={pr.mean():.4g} median={np.median(pr):.4g} std={pr.std():.4g}")
finite = np.isfinite(pr).all()
print(f"all finite={finite}  n_at_floor(<=1e-5)={(pr<=1e-5).sum()}  n_equal_max={(pr==pr.max()).sum()}")
# alpha-weighted sampling probs
alpha = cfg.per_alpha
probs = (np.where(np.isfinite(pr)&(pr>0), pr, 1.0)) ** alpha
probs /= probs.sum()
# effective sample size = how concentrated sampling is. N => uniform.
ess = 1.0 / (probs**2).sum()
print(f"alpha={alpha}  ESS={ess:.1f} / {len(pr)}  (=N means uniform; <<N means concentrated)")
print(f"  top-5 prob games take {np.sort(probs)[-5:].sum()*100:.2f}% of sampling mass")
qs = np.quantile(pr, [0.0,0.1,0.25,0.5,0.75,0.9,0.99,1.0])
print(f"  priority quantiles [0,.1,.25,.5,.75,.9,.99,1]: {np.array2string(qs, precision=4)}")

# ---- 2. GAME COMPOSITION (decisive / warmstart) ----
print("\n--- BUFFER COMPOSITION ---")
n_warm = sum(1 for g in rb.buffer if g.external_values)
outcomes = np.array([g.game_outcome for g in rb.buffer])
n_dec = sum(1 for g in rb.buffer if abs(g.game_outcome) >= 0.5 and not g.external_values)
n_draw_sp = sum(1 for g in rb.buffer if g.game_outcome == 0.0 and not g.external_values)
print(f"warmstart={n_warm}  decisive_selfplay={n_dec}  draw_selfplay={n_draw_sp}")
print(f"outcome counts: +1={(outcomes>0).sum()} -1={(outcomes<0).sum()} 0={(outcomes==0).sum()}")
lens = np.array([len(g) for g in rb.buffer])
print(f"game lengths: min={lens.min()} max={lens.max()} mean={lens.mean():.1f}")
rep_flags = sum(1 for g in rb.buffer if g.draw_by_repetition)
nop_flags = sum(1 for g in rb.buffer if g.draw_by_no_progress)
print(f"draw_by_repetition={rep_flags}  draw_by_no_progress={nop_flags}")

# ---- 3. SAMPLE A REAL BATCH ----
print("\n--- SAMPLE_BATCH ---")
beta = cfg.per_beta_init + (1.0 - cfg.per_beta_init) * (int(step) / max(1, cfg.training_steps)) if str(step).isdigit() else cfg.per_beta_init
print(f"beta(annealed at step {step}) = {beta:.4f}")
batch, gidx, isw = rb.sample_batch(
    cfg.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount, game.action_space_size,
    alpha=cfg.per_alpha, beta=beta, value_head_type=cfg.value_head_type,
    history_frames=cfg.history_frames, eval_to_wdl_alpha=cfg.eval_to_wdl_alpha,
    eval_to_wdl_beta=cfg.eval_to_wdl_beta, warmstart_sample_frac=cfg.warmstart_sample_frac,
    decisive_sample_frac=cfg.decisive_sample_frac, q_ratio=cfg.q_ratio,
    warmstart_q_ratio=cfg.warmstart_q_ratio, selfplay_q_ratio=cfg.selfplay_q_ratio,
    repetition_penalty=cfg.repetition_penalty,
    repetition_penalty_window=int(getattr(cfg,"repetition_penalty_window",0)),
    repetition_penalty_decay=float(cfg.repetition_penalty_decay),
    build_legal_masks=bool(cfg.mask_illegal_policy),
)
print(f"sampled {len(gidx)} positions from {len(set(gidx.tolist()))} distinct games")
# decisive fraction in the SAMPLED batch
samp_dec = sum(1 for i in gidx if abs(rb.buffer[i].game_outcome) >= 0.5 and not rb.buffer[i].external_values)
samp_warm = int(batch["is_warmstart"].sum())
print(f"sampled-batch decisive_selfplay={samp_dec}/{len(gidx)} ({samp_dec/len(gidx)*100:.0f}%)  warmstart={samp_warm}")
print(f"IS weights: min={isw.min():.4f} max={isw.max():.4f} mean={isw.mean():.4f} (max should be 1.0 after normalization)")

# ---- 4. TARGET VALUE (WDL) SHAPE / NORMALIZATION ----
tv = batch["target_values"]  # (B, K+1, 3) for wdl
print(f"\n--- TARGET_VALUES (wdl) shape={tuple(tv.shape)} ---")
sums = tv.sum(dim=-1)
print(f"row-sum over WDL: min={sums.min():.5f} max={sums.max():.5f} (should be ~1.0)")
tv0 = tv[:,0]  # root WDL
print(f"root WDL means: P_W={tv0[:,0].mean():.3f} P_D={tv0[:,1].mean():.3f} P_L={tv0[:,2].mean():.3f}")
# scalar target V = W - L
scal = (tv0[:,0] - tv0[:,2])
print(f"root scalar target V=W-L: mean={scal.mean():.4f} std={scal.std(unbiased=False):.4f}  |V|>=0.5 frac={(scal.abs()>=0.5).float().mean():.3f}")
# how many root targets are essentially draw [0,1,0] vs decisive?
near_draw = ((tv0[:,1] > 0.6)).float().mean()
print(f"root targets with P_D>0.6 (drawish): {near_draw*100:.1f}%")

# ---- 5. RUN THE ACTUAL TRAIN STEP via Trainer._train_step ----
print("\n--- ACTUAL _train_step LOSS DECOMPOSITION ---")
from src.training.trainer import MuZeroTrainer
import tempfile
tr = MuZeroTrainer(cfg, game, net, run_id="probe", device="cpu",
                   log_dir=tempfile.mkdtemp(), checkpoints_dir=tempfile.mkdtemp())
tr.replay_buffer = rb
tr.global_step = int(step) if str(step).isdigit() else 0
# disable AMP autocast on CPU (config has it True but CPU autocast is bf16; keep faithful but cpu)
info = tr._train_step()
keys = ["total_loss","policy_loss","value_loss","reward_loss","consistency_loss",
        "inverse_loss","moves_left_loss","value_loss_weight","value_mae","value_target_std",
        "policy_entropy_pred","policy_entropy_target","grad_norm",
        "value_loss_warmstart","value_loss_selfplay","policy_loss_warmstart","policy_loss_selfplay",
        "batch_warmstart_frac"]
for k in keys:
    v = info.get(k, None)
    print(f"  {k:28s} = {v}")

# manual weighting cross-check: what fraction of total_loss is the value term?
K = cfg.num_unroll_steps
outer = 1.0/(K+1)
print(f"\n  outer_scale=1/(K+1)={outer:.4f}  value_loss_weight(applied)={info.get('value_loss_weight')}")
print(f"  -> value contribution to total ~= value_loss * weight = {info.get('value_loss')} (already includes outer)")
print(f"  -> displayed value_loss DROPS the per-sample weight; weighted contribution = value_loss*weight = "
      f"{info.get('value_loss')*info.get('value_loss_weight'):.5f}")
