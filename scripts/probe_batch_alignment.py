"""Verify sample_batch's stacked tensors preserve per-game make_target alignment,
and that the trainer's slicing (actions[:,k], target_*[:, k+1]) lines up.

Monkeypatches np.random.randint so we KNOW which (game, pos) each batch row used,
then re-derives the expected per-row targets directly from make_target and
asserts the batch tensor equals them.
CPU only.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

np.random.seed(0)
torch.manual_seed(0)

BUF = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"

cfg = get_config("chess_small")
cfg.device = "cpu"
game = ChessGame()
A = game.action_space_size
K = cfg.num_unroll_steps
HF = cfg.history_frames

buf = ReplayBuffer(max_size=10**9)
buf.load(BUF, game=ChessGame())
print(f"loaded {len(buf)} games")

# Force flat PER sampling (no stratification) so game_indices come straight from
# np.random.choice and pos from np.random.randint — both deterministic under seed.
cfg.warmstart_sample_frac = 0.0
cfg.decisive_sample_frac = 0.0   # disable stratification to make pos predictable

# Capture the (game_idx, pos) pairs sample_batch draws by wrapping randint.
captured_pos = []
orig_randint = np.random.randint
def spy_randint(low, high=None, *a, **kw):
    r = orig_randint(low, high, *a, **kw)
    # sample_batch calls randint(0, len(game)) once per game (scalar).
    if high is not None and np.isscalar(r):
        captured_pos.append(int(r))
    return r
np.random.seed(123)
np.random.randint = spy_randint

batch, game_indices, weights = buf.sample_batch(
    batch_size=16, num_unroll_steps=K, td_steps=cfg.td_steps,
    discount=cfg.discount, action_space_size=A,
    alpha=cfg.per_alpha, beta=1.0,
    value_head_type=cfg.value_head_type, history_frames=HF,
    eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
    warmstart_sample_frac=0.0, decisive_sample_frac=0.0,
    q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
    selfplay_q_ratio=cfg.selfplay_q_ratio,
    repetition_penalty=cfg.repetition_penalty,
    repetition_penalty_window=int(cfg.repetition_penalty_window),
    repetition_penalty_decay=float(cfg.repetition_penalty_decay),
    build_legal_masks=bool(getattr(cfg, "mask_illegal_policy", False)),
)
np.random.randint = orig_randint

print(f"game_indices={list(game_indices)}")
print(f"captured_pos={captured_pos}  (len={len(captured_pos)})")
print(f"batch tensor shapes:")
for k, v in batch.items():
    if torch.is_tensor(v):
        print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")

# captured_pos should be aligned 1:1 with game_indices (one randint per game).
assert len(captured_pos) == len(game_indices), \
    f"randint capture mismatch: {len(captured_pos)} vs {len(game_indices)}"

# Re-derive each row's targets directly from make_target and compare.
n_rows = len(game_indices)
fails = 0
checks = 0
for row in range(n_rows):
    g = buf.buffer[game_indices[row]]
    pos = captured_pos[row]
    obs_list, obs_mask, actions, values, rewards, policies = g.make_target(
        pos, K, cfg.td_steps, cfg.discount, A,
        value_head_type=cfg.value_head_type, history_frames=HF,
        eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
        q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
        selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=cfg.repetition_penalty,
        repetition_penalty_window=int(cfg.repetition_penalty_window),
        repetition_penalty_decay=float(cfg.repetition_penalty_decay),
    )
    # actions
    if not torch.equal(batch["actions"][row], torch.tensor(actions, dtype=torch.long)):
        print(f"row {row}: ACTION MISMATCH batch={batch['actions'][row].tolist()} exp={actions}")
        fails += 1
    checks += 1
    # rewards
    if not torch.allclose(batch["target_rewards"][row], torch.tensor(rewards, dtype=torch.float32)):
        print(f"row {row}: REWARD MISMATCH")
        fails += 1
    checks += 1
    # values (WDL: (K+1,3))
    exp_vals = torch.from_numpy(np.asarray(values, dtype=np.float32))
    if not torch.allclose(batch["target_values"][row], exp_vals):
        print(f"row {row}: VALUE MISMATCH")
        fails += 1
    checks += 1
    # policies
    exp_pol = torch.from_numpy(np.stack(policies))
    if not torch.allclose(batch["target_policies"][row], exp_pol):
        print(f"row {row}: POLICY MISMATCH")
        fails += 1
    checks += 1
    # obs_mask
    if not torch.allclose(batch["target_obs_mask"][row], torch.tensor(obs_mask, dtype=torch.float32)):
        print(f"row {row}: OBS_MASK MISMATCH")
        fails += 1
    checks += 1
    # observations root (k=0)
    if not torch.allclose(batch["observations"][row], obs_list[0]):
        print(f"row {row}: ROOT OBS MISMATCH")
        fails += 1
    checks += 1

print(f"\nbatch<->make_target re-derivation: {checks-fails}/{checks} checks PASS, {fails} FAIL")

# ---- Trainer-slicing semantics: at unroll step k, actions[:,k] applied to
# h_k predicts reward target_rewards[:,k+1] and lands in state with value
# target_values[:,k+1]/policy target_policies[:,k+1]. Confirm shapes line up. ----
print("\nTrainer slice shapes:")
print(f"  actions[:, k] for k in 0..{K-1}: actions has K={batch['actions'].shape[1]} cols")
print(f"  target_values[:, k+1] for k in 0..{K-1}: values has K+1={batch['target_values'].shape[1]} cols")
print(f"  target_rewards[:, k+1]: rewards has {batch['target_rewards'].shape[1]} cols")
print(f"  target_policies[:, k+1]: policies has {batch['target_policies'].shape[1]} cols")
assert batch["actions"].shape[1] == K
assert batch["target_values"].shape[1] == K + 1
assert batch["target_rewards"].shape[1] == K + 1
assert batch["target_policies"].shape[1] == K + 1
print("  slice-shape invariants: PASS")
