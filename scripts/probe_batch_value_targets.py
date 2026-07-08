"""What value/policy targets does the trainer ACTUALLY see, post-sampling?
Reproduce sample_batch with the live config (incl decisive_sample_frac=0.5) and
report the distribution of value-target scalars (W-L), the fraction of decisive
vs draw targets, and policy-target sharpness. This shows whether the
decisive_sample_frac sampler delivers a non-constant value signal.
CPU only.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
cfg = get_config("chess_small"); cfg.device="cpu"
game = ChessGame(); A = game.action_space_size
buf = ReplayBuffer(max_size=10**9); buf.load(BUF, game=ChessGame())

np.random.seed(1); torch.manual_seed(1)
B = 256
# Aggregate over several batches to get a stable picture.
all_v0 = []        # root value-target scalar (W-L)
all_vk = []        # all unroll value targets (masked-in only)
decisive_root = 0  # |W-L| > 0.5 at root
total_root = 0
pol_max = []       # max prob of root policy target
pol_nnz = []
n_warm = 0
for it in range(20):
    batch, gidx, w = buf.sample_batch(
        B, cfg.num_unroll_steps, cfg.td_steps, cfg.discount, A,
        alpha=cfg.per_alpha, beta=1.0, value_head_type=cfg.value_head_type,
        history_frames=cfg.history_frames,
        decisive_sample_frac=cfg.decisive_sample_frac,
        selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=cfg.repetition_penalty,
        repetition_penalty_decay=float(cfg.repetition_penalty_decay))
    tv = batch["target_values"]              # (B, K+1, 3)
    mask = batch["target_obs_mask"]          # (B, K+1)
    v_wl = (tv[..., 0] - tv[..., 2])         # (B, K+1)
    all_v0.append(v_wl[:, 0].numpy())
    m = mask.bool()
    all_vk.append(v_wl[m].numpy())
    decisive_root += int((v_wl[:, 0].abs() > 0.5).sum())
    total_root += B
    tp = batch["target_policies"][:, 0]      # (B, A)
    pol_max.append(tp.max(dim=1).values.numpy())
    pol_nnz.append((tp > 0).sum(dim=1).numpy())
    n_warm += int(batch["is_warmstart"].sum())

v0 = np.concatenate(all_v0); vk = np.concatenate(all_vk)
pmax = np.concatenate(pol_max); pnnz = np.concatenate(pol_nnz)
print(f"=== ROOT value targets (W-L scalar), {len(v0)} samples, decisive_sample_frac={cfg.decisive_sample_frac} ===")
print(f"  mean={v0.mean():+.4f} std={v0.std():.4f}")
hist = {f'{lo:+.2f}..{hi:+.2f}': int(((v0>=lo)&(v0<hi)).sum())
        for lo,hi in [(-1.01,-0.5),(-0.5,-0.05),(-0.05,0.05),(0.05,0.5),(0.5,1.01)]}
print(f"  histogram: {hist}")
print(f"  frac |W-L|>0.5 (decisive root): {decisive_root/total_root:.3f}  (target ~0.5)")
print(f"  frac exactly draw-ish (-0.05..0.05): {np.mean(np.abs(v0)<0.05):.3f}")
print(f"\n=== ALL unroll value targets (masked-in), {len(vk)} samples ===")
print(f"  mean={vk.mean():+.4f} std={vk.std():.4f}  frac|.|>0.5={np.mean(np.abs(vk)>0.5):.3f}")
print(f"\n=== ROOT policy targets ===")
print(f"  max-prob: mean={pmax.mean():.3f} median={np.median(pmax):.3f}  (1.0=one-hot)")
print(f"  nnz: mean={pnnz.mean():.1f} median={np.median(pnnz):.0f}")
print(f"\n  warmstart fraction in batches: {n_warm/(20*B):.3f}")
