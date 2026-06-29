"""Cross-check the THREE history-stacking paths produce identical channel order:
 1. GPU self-play: obs_window roll (self_play.py:506-509)
 2. inference-time: stack_with_history (replay_buffer.py:458)
 3. training-time:  GameHistory._stack_history (replay_buffer.py:175)
A mismatch = the net sees different frame order at train vs inference (silent).
CPU only, synthetic per-frame tensors so we can track identity.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/chessZero")
from src.training.replay_buffer import GameHistory, stack_with_history

NP = 3      # fake per-frame planes
T = 4       # history_frames
H = W = 2
# distinct per-ply observations: frame f has all-value f
nply = 6
obs = [torch.full((NP, H, W), float(f)) for f in range(nply)]

# ---- 1. GPU self-play obs_window roll, at each ply capture the stacked obs ----
obs_window = torch.zeros(1, T, NP, H, W)
gpu_stacks = []
for f in range(nply):
    obs_window = torch.roll(obs_window, shifts=1, dims=1)
    obs_window[:, 0] = obs[f].unsqueeze(0)
    stacked = obs_window.reshape(1, T*NP, H, W)[0]
    gpu_stacks.append(stacked)

# ---- 2. inference-time stack_with_history (current=obs[f], prior=obs[:f]) ----
inf_stacks = []
for f in range(nply):
    s = stack_with_history(obs[f], obs[:f], T)
    inf_stacks.append(s)

# ---- 3. training-time _stack_history ----
g = GameHistory()
g.observations = obs
train_stacks = [g._stack_history(f, T) for f in range(nply)]

print(f"{'ply':>3} {'GPU frame-values (per slot)':>32} {'inf':>20} {'train':>20}")
def slotvals(stk):
    return [float(stk[t*NP:(t+1)*NP].flatten()[0]) for t in range(T)]
all_ok = True
for f in range(nply):
    gv, iv, tv = slotvals(gpu_stacks[f]), slotvals(inf_stacks[f]), slotvals(train_stacks[f])
    ok = (gv == iv == tv)
    all_ok &= ok
    print(f"{f:>3} {str(gv):>32} {str(iv):>20} {str(tv):>20}  {'OK' if ok else 'MISMATCH'}")
print(f"\nAll three stacking paths agree on channel order: {'PASS' if all_ok else 'FAIL'}")
print("(slot 0 = newest frame; should equal current ply value, then decreasing)")
