"""Probe the zero-padding + opening region of the 8-frame stack.

The early-game stack is where frame-order/off-by-one bugs hide: at ply 0 the
window is [obs0, 0,0,0,0,0,0,0]; at ply 3 it is [obs3,obs2,obs1,obs0,0,0,0,0].
This compares the GPU self-play window against _stack_history(idx) for the FIRST
12 plies specifically, asserting the zero-pad boundary lands in the same channel
slots in both paths. Also checks the random-opening region (n_random=8) where
self-play uses uniform sampling, not MCTS, but still rolls the window.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.training.replay_buffer import GameHistory

N_FRAMES = 8
device = "cuda"
gpu = GpuChessGame(); gpu.max_plies = 60
cpu = ChessGame()
c = gpu.num_planes

state = gpu.reset_batch(1, device=device)
next_legal_mask = gpu.legal_mask(state)
obs_window = torch.zeros(1, N_FRAMES, c, 8, 8, device=device, dtype=torch.float32)

captured, single_obs, actions = [], [], []
rng = np.random.default_rng(7)

for ply in range(24):
    so = gpu.to_tensor_batch(state)
    obs_window = torch.roll(obs_window, shifts=1, dims=1)
    obs_window[:, 0] = so
    stacked = obs_window.reshape(1, N_FRAMES * c, 8, 8)
    legal = next_legal_mask[0].cpu().numpy().nonzero()[0]
    a = int(rng.choice(legal))
    captured.append(stacked[0].cpu().clone())
    single_obs.append(so[0].cpu().clone())
    actions.append(a)
    at = torch.tensor([a], dtype=torch.int64, device=device)
    state, _, _, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        break

L = len(actions)
gh = GameHistory(game_name="chess")
for t in range(L):
    gh.observations.append(single_obs[t])
    gh.actions.append(int(actions[t]))
    p = np.zeros(cpu.action_space_size, dtype=np.float32); p[int(actions[t])] = 1.0
    gh.policies.append(p); gh.root_values.append(0.0); gh.rewards.append(0.0)
gh.game_outcome = 0
gh.observations.append(single_obs[-1])
recon = GameHistory.from_compact_dict(gh.to_compact_dict(), ChessGame())

print(f"L={L}. Checking zero-pad boundary per ply (frame slot where it goes to 0):")
for t in range(min(L, 12)):
    saw = captured[t].view(N_FRAMES, c, 8, 8)
    trn = recon._stack_history(t, N_FRAMES).view(N_FRAMES, c, 8, 8)
    # Find first all-zero frame slot in each (the pad boundary). Use piece planes
    # (0..11): a real position has >=2 kings so piece sum > 0; padding is all 0.
    def pad_start(stack):
        for f in range(N_FRAMES):
            if float(stack[f, 0:12].abs().sum()) == 0.0:
                return f
        return N_FRAMES
    ps_saw, ps_trn = pad_start(saw), pad_start(trn)
    diff = float((saw - trn).abs().max())
    flag = "" if (ps_saw == ps_trn and diff < 1e-5) else "  <<< MISMATCH"
    print(f"  ply {t:2d}: pad_start saw={ps_saw} trained={ps_trn} | "
          f"max_diff={diff:.5f}{flag}")

worst = max(float((captured[t] - recon._stack_history(t, N_FRAMES)).abs().max())
            for t in range(L))
print(f"\nWorst diff over all {L} plies: {worst:.6f}")
