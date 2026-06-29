"""Trace DRAW games (the 99% majority) + repetition-penalty target path.
Also: policy normalization, root_value POV sanity, full-batch sample_batch shape/index check.
CPU only.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

BUF = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
cfg = get_config("chess_small"); cfg.device = "cpu"
game = ChessGame(); A = game.action_space_size
buf = ReplayBuffer(max_size=10**9); buf.load(BUF, game=ChessGame())

# stats over buffer
outs = np.array([g.game_outcome for g in buf.buffer])
reps = np.array([g.draw_by_repetition for g in buf.buffer])
nps = np.array([g.draw_by_no_progress for g in buf.buffer])
lens = np.array([len(g.actions) for g in buf.buffer])
print(f"games={len(buf)}  decisive={(outs!=0).sum()}  draw_rep={reps.sum()}  "
      f"draw_noprog={nps.sum()}  other_draw={((outs==0)&~reps&~nps).sum()}")
print(f"len: mean={lens.mean():.1f} min={lens.min()} max={lens.max()} "
      f"median={np.median(lens):.0f}")

# policy normalization across ALL plies of all games
sums, nzs = [], []
for g in buf.buffer:
    for p in g.policies:
        if p is not None and len(p):
            sums.append(float(p.sum())); nzs.append(int((p>0).sum()))
sums = np.array(sums); nzs = np.array(nzs)
print(f"\npolicy sum: mean={sums.mean():.4f} min={sums.min():.4f} max={sums.max():.4f} "
      f"frac!=1(>1e-3)={np.mean(np.abs(sums-1)>1e-3):.4f}")
print(f"policy nnz: mean={nzs.mean():.1f} min={nzs.min()} max={nzs.max()} "
      f"frac_onehot={np.mean(nzs==1):.4f}")

# root_value POV / range
rv = np.concatenate([np.array(g.root_values) for g in buf.buffer if g.root_values])
print(f"\nroot_value: mean={rv.mean():+.4f} std={rv.std():.4f} "
      f"min={rv.min():+.3f} max={rv.max():+.3f}  |rv|>0.25 frac={np.mean(np.abs(rv)>0.25):.4f}")

# ---- repetition-penalty target inspection on a draw_by_rep game ----
rep_idx = [i for i,g in enumerate(buf.buffer) if g.draw_by_repetition]
print(f"\ndraw_by_repetition games: {len(rep_idx)}")
if rep_idx:
    g = buf.buffer[rep_idx[0]]; N = len(g.actions)
    si = max(0, N - 4)
    _,_,_,values,_,_ = g.make_target(
        si, cfg.num_unroll_steps, cfg.td_steps, cfg.discount, A,
        value_head_type=cfg.value_head_type, history_frames=cfg.history_frames,
        eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
        q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
        selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=cfg.repetition_penalty,
        repetition_penalty_window=int(cfg.repetition_penalty_window),
        repetition_penalty_decay=float(cfg.repetition_penalty_decay))
    print(f"  game {rep_idx[0]} N={N} outcome={g.game_outcome} rep_end_idx={N}")
    print(f"  repetition_penalty={cfg.repetition_penalty} decay={cfg.repetition_penalty_decay}")
    print("  WDL value targets near end (expect D->L tilt growing toward terminal):")
    for i in range(cfg.num_unroll_steps + 1):
        idx = si + i
        v = values[i]
        plies_to_end = (N - idx)
        exp_w = cfg.repetition_penalty * (cfg.repetition_penalty_decay ** plies_to_end) if idx < N else None
        rv_here = g.root_values[idx] if idx < len(g.root_values) else None
        print(f"    i={i} idx={idx} plies_to_end={plies_to_end} root_v={rv_here} "
              f"WDL={np.round(v,4).tolist() if isinstance(v,np.ndarray) else v}  "
              f"expected_L(d)={None if exp_w is None else round(exp_w,4)}")

# ---- full sample_batch shape + per-sample index round-trip ----
print("\n=== sample_batch shapes ===")
np.random.seed(0)
batch, gidx, w = buf.sample_batch(
    8, cfg.num_unroll_steps, cfg.td_steps, cfg.discount, A,
    alpha=cfg.per_alpha, beta=1.0, value_head_type=cfg.value_head_type,
    history_frames=cfg.history_frames, decisive_sample_frac=cfg.decisive_sample_frac,
    selfplay_q_ratio=cfg.selfplay_q_ratio, repetition_penalty=cfg.repetition_penalty,
    repetition_penalty_decay=float(cfg.repetition_penalty_decay))
for k,v in batch.items():
    print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
print(f"  game_indices={gidx}")
# verify obs[:,0] == obs root and target_observations[:,0]
assert torch.equal(batch["observations"], batch["target_observations"][:,0])
print("  observations == target_observations[:,0]: OK")
# verify per-frame channel count = num_planes * history_frames
print(f"  obs channels={batch['observations'].shape[1]} "
      f"expect num_planes*history_frames={game.num_planes}*{cfg.history_frames}="
      f"{game.num_planes*cfg.history_frames}")
