"""Trace ONE real buffer game end-to-end through make_target → sample_batch.

Verifies obs[t] / action[t] / value-target[t] / policy-target[t] / reward-target[t]
allocation + alignment, the K-unroll offset, STM parity, and reward placement.
CPU only.
"""
import sys, pickle
import numpy as np
import torch

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory, ReplayBuffer

BUF = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"

cfg = get_config("chess_small")
cfg.device = "cpu"
game = ChessGame()
A = game.action_space_size
print(f"action_space_size={A}  value_head_type={cfg.value_head_type}  td_steps={cfg.td_steps}")
print(f"K(num_unroll_steps)={cfg.num_unroll_steps}  history_frames={cfg.history_frames}")
print(f"selfplay_q_ratio={cfg.selfplay_q_ratio}  rep_penalty={cfg.repetition_penalty} decay={cfg.repetition_penalty_decay}")

# Load buffer (v3 compact needs a fresh game per record).
buf = ReplayBuffer(max_size=10**9)
buf.load(BUF, game=ChessGame())
print(f"\nLoaded {len(buf)} games from {BUF}")

# Find a DECISIVE game if any (most interesting for value parity); else first.
dec = [i for i, g in enumerate(buf.buffer) if g.game_outcome != 0]
print(f"decisive games: {len(dec)}/{len(buf)}")
pick = dec[0] if dec else 0
g: GameHistory = buf.buffer[pick]
print(f"\n=== GAME {pick}: len(obs)={len(g.observations)} len(actions)={len(g.actions)} "
      f"len(rewards)={len(g.rewards)} len(policies)={len(g.policies)} "
      f"len(root_values)={len(g.root_values)} ===")
print(f"game_outcome(white POV)={g.game_outcome}  draw_by_rep={g.draw_by_repetition} "
      f"draw_by_noprog={g.draw_by_no_progress}  reanalyze_count={g.reanalyze_count}")
print(f"nonzero rewards at plies: {[t for t,r in enumerate(g.rewards) if r!=0]} "
      f"values={[g.rewards[t] for t in range(len(g.rewards)) if g.rewards[t]!=0]}")

# Structural invariants
assert len(g.observations) == len(g.actions) + 1, "obs must be N+1"
for nm in ("rewards", "policies", "root_values", "legal_actions_list"):
    L = len(getattr(g, nm))
    print(f"  len({nm})={L} {'OK' if L==len(g.actions) else 'MISMATCH'}")

# Print a window near the end of the game (where the decisive move/outcome lives).
N = len(g.actions)
lo = max(0, N - 6)
print(f"\n--- per-ply table (plies {lo}..{N-1}); STM: even=White, odd=Black ---")
print(f"{'ply':>4} {'STM':>5} {'action':>7} {'reward':>7} {'root_v(STM)':>11} "
      f"{'pol_argmax':>10} {'pol@action':>10}")
for t in range(lo, N):
    stm = "W" if t % 2 == 0 else "B"
    pa = int(np.argmax(g.policies[t])) if g.policies[t] is not None and len(g.policies[t]) else -1
    pat = float(g.policies[t][g.actions[t]]) if g.policies[t] is not None and len(g.policies[t]) else 0.0
    print(f"{t:>4} {stm:>5} {g.actions[t]:>7} {g.rewards[t]:>7.2f} {g.root_values[t]:>11.4f} "
          f"{pa:>10} {pat:>10.4f}")

# ---- make_target trace at a state_index near the terminal so we see the outcome land ----
def wdl_to_scalar(v):
    if isinstance(v, np.ndarray) and v.shape == (3,):
        return float(v[0] - v[2])  # W - L
    return float(v)

K = cfg.num_unroll_steps
# Choose state_index so the terminal (decisive) ply is inside the unroll window.
si = max(0, N - 2)
obs_list, obs_mask, actions, values, rewards, policies = g.make_target(
    si, K, cfg.td_steps, cfg.discount, A,
    value_head_type=cfg.value_head_type,
    history_frames=cfg.history_frames,
    eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
    q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
    selfplay_q_ratio=cfg.selfplay_q_ratio,
    repetition_penalty=cfg.repetition_penalty,
    repetition_penalty_window=int(cfg.repetition_penalty_window),
    repetition_penalty_decay=float(cfg.repetition_penalty_decay),
)
print(f"\n=== make_target(state_index={si}, K={K}) ===")
print(f"len(obs_list)={len(obs_list)} obs_mask={obs_mask}")
print(f"target_actions(K)={actions}")
print(f"{'i':>3} {'idx':>4} {'mask':>5} {'STM':>4} {'val(W-L)':>9} {'val_raw':>22} "
      f"{'rew':>6} {'polargmax':>9}")
for i in range(K + 1):
    idx = si + i
    stm = "W" if idx % 2 == 0 else "B"
    v = values[i]
    pa = int(np.argmax(policies[i])) if policies[i] is not None and len(policies[i]) else -1
    vraw = (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else round(float(v), 4))
    print(f"{i:>3} {idx:>4} {obs_mask[i]:>5.0f} {stm:>4} {wdl_to_scalar(v):>9.3f} "
          f"{str(vraw):>22} {rewards[i]:>6.2f} {pa:>9}")

# ---- ALIGNMENT ASSERTIONS the trainer relies on ----
print("\n=== ALIGNMENT CHECKS (trainer reads target_*[:, k+1] at unroll step k) ===")
# 1. reward: target_rewards[k+1] must equal self.rewards[si+k]
ok_r = True
for k in range(K):
    want = g.rewards[si + k] if (si + k) < len(g.rewards) else 0.0
    got = rewards[k + 1]
    aligned = abs(got - want) < 1e-6
    ok_r &= aligned
    flag = "" if aligned else "  <-- MISALIGNED"
    print(f"  reward k={k}: target_rewards[{k+1}]={got:.3f}  self.rewards[{si+k}]="
          f"{want:.3f}{flag}")
print(f"  reward alignment: {'PASS' if ok_r else 'FAIL'}")

# 2. action: target_actions[k] must equal self.actions[si+k]
ok_a = True
for k in range(K):
    want = g.actions[si + k] if (si + k) < len(g.actions) else 0
    got = actions[k]
    aligned = got == want
    ok_a &= aligned
    print(f"  action k={k}: target_actions[{k}]={got}  self.actions[{si+k}]={want}"
          f"{'' if aligned else '  <-- MISALIGNED'}")
print(f"  action alignment: {'PASS' if ok_a else 'FAIL'}")

# 3. policy: target_policies[i] must equal self.policies[si+i] (for valid idx)
ok_p = True
for i in range(K + 1):
    idx = si + i
    if idx < len(g.policies) and g.policies[idx] is not None and len(g.policies[idx]):
        want = int(np.argmax(g.policies[idx]))
        got = int(np.argmax(policies[i])) if policies[i].sum() > 0 else -1
        aligned = (got == want) or (policies[i].sum() == 0 and idx >= len(g.policies))
        if policies[i].sum() == 0 and idx < len(g.actions):
            aligned = False  # should not be zero inside the game
        ok_p &= aligned
        print(f"  policy i={i} idx={idx}: argmax(target)={got} argmax(stored)={want}"
              f"{'' if aligned else '  <-- MISALIGNED'}")
print(f"  policy alignment: {'PASS' if ok_p else 'FAIL'}")

# 4. value STM-parity sanity: for a decisive game, the winner's plies should
#    get positive W-L value, loser's negative — at the terminal index.
if g.game_outcome != 0:
    term = N  # terminal observation index
    # value at idx==N (terminal) is past-actions but inside obs; check make_target
    # from si covering it:
    print("\n=== VALUE STM-PARITY (decisive game) ===")
    for i in range(K + 1):
        idx = si + i
        if idx >= len(g):  # past end
            continue
        stm_is_white = (idx % 2 == 0)
        stm_won = (g.game_outcome > 0) == stm_is_white
        sval = wdl_to_scalar(values[i])
        sign_ok = (sval > 0) == stm_won if abs(sval) > 1e-3 else True
        print(f"  idx={idx} STM={'W' if stm_is_white else 'B'} stm_won={stm_won} "
              f"val(W-L)={sval:+.3f} parity={'OK' if sign_ok else 'WRONG-SIGN'}")
