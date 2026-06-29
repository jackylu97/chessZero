"""Probe: does the obs the net SEES in self-play match the obs it TRAINS on?

Drives a real game through the GPU self-play env (chess_gpu), capturing the
exact stacked observation passed to MCTS at each ply (the 8-frame, newest-first
stack). Then reconstructs the same game through the training path
(GameHistory.from_compact_dict -> CPU ChessGame.to_tensor + _stack_history) and
compares channel-for-channel.

A mismatch means the value head is trained on different input than it played
with -> it cannot learn position values.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.training.replay_buffer import GameHistory, stack_with_history

torch.manual_seed(0)
np.random.seed(0)

N_FRAMES = 8
device = "cuda"

gpu = GpuChessGame()
gpu.max_plies = 300
cpu = ChessGame()

NUM_GAMES = 4
state = gpu.reset_batch(NUM_GAMES, device=device)

c, h, w = gpu.num_planes, 8, 8
obs_window = torch.zeros(NUM_GAMES, N_FRAMES, c, h, w, device=device, dtype=torch.float32)

# Per-game captured stacks: list over ply -> [N, n_frames*c, h, w] (the input to MCTS)
captured_stacks = [[] for _ in range(NUM_GAMES)]
single_obs_per_ply = [[] for _ in range(NUM_GAMES)]  # what gets stored to GameHistory
actions_per_ply = [[] for _ in range(NUM_GAMES)]

next_legal_mask = gpu.legal_mask(state)
alive = torch.ones(NUM_GAMES, dtype=torch.bool, device=device)
game_outcome = torch.zeros(NUM_GAMES, dtype=torch.int32, device=device)

rng = np.random.default_rng(12345)

for ply in range(300):
    single_obs = gpu.to_tensor_batch(state)              # [N, C, H, W]
    legal_mask = next_legal_mask
    any_legal = legal_mask.any(dim=1, keepdim=True)

    # roll window, newest at slot 0 (EXACT copy of self_play.py logic)
    obs_window = torch.roll(obs_window, shifts=1, dims=1)
    obs_window[:, 0] = single_obs
    stacked_obs = obs_window.reshape(NUM_GAMES, N_FRAMES * c, h, w)

    # Pick a deterministic legal action per alive game (no MCTS needed for obs test)
    legal_cpu = legal_mask.cpu().numpy()
    actions = np.zeros(NUM_GAMES, dtype=np.int64)
    for g in range(NUM_GAMES):
        if not bool(alive[g]):
            continue
        legal_idx = np.nonzero(legal_cpu[g])[0]
        if len(legal_idx) == 0:
            actions[g] = 0
            continue
        actions[g] = int(rng.choice(legal_idx))
        captured_stacks[g].append(stacked_obs[g].cpu().clone())
        single_obs_per_ply[g].append(single_obs[g].cpu().clone())
        actions_per_ply[g].append(actions[g])

    action_t = torch.tensor(actions, dtype=torch.int64, device=device)
    state, rewards, _, next_legal_mask = gpu.step_batch_with_legal(state, action_t)

    newly_done = state.done & alive
    game_outcome = torch.where(newly_done, state.winner.to(torch.int32), game_outcome)
    alive = alive & ~state.done
    if not bool(alive.any()):
        break

print(f"Games finished at ply {ply}. Lengths:",
      [len(a) for a in actions_per_ply])

# Now reconstruct via the TRAINING path for each game and compare.
worst_overall = 0.0
n_plane_mismatch_total = 0
per_plane_max = np.zeros(c)
for g in range(NUM_GAMES):
    L = len(actions_per_ply[g])
    if L < 4:
        continue
    # Build a GameHistory the same way self_play.py would, then round-trip
    # through to_compact_dict / from_compact_dict (the actual disk path).
    gh = GameHistory(game_name="chess")
    for t in range(L):
        gh.observations.append(single_obs_per_ply[g][t])
        gh.actions.append(int(actions_per_ply[g][t]))
        p = np.zeros(cpu.action_space_size, dtype=np.float32)
        p[int(actions_per_ply[g][t])] = 1.0
        gh.policies.append(p)
        gh.root_values.append(0.0)
        gh.rewards.append(0.0)
    gh.game_outcome = int(game_outcome[g].item())
    # final obs
    gh.observations.append(single_obs_per_ply[g][-1])  # placeholder, not compared

    compact = gh.to_compact_dict()
    recon = GameHistory.from_compact_dict(compact, ChessGame())

    # Compare at each ply: the stack the net SAW (captured) vs the stack the
    # trainer would build via _stack_history(idx, N_FRAMES).
    for t in range(L):
        saw = captured_stacks[g][t]                       # [n_frames*c, h, w]
        trained = recon._stack_history(t, N_FRAMES)       # [n_frames*c, h, w]
        diff = (saw - trained).abs()
        m = float(diff.max())
        worst_overall = max(worst_overall, m)
        if m > 1e-4:
            n_plane_mismatch_total += 1
            # which planes differ?
            per_frame = diff.view(N_FRAMES, c, h, w)
            plane_max = per_frame.amax(dim=(0, 2, 3)).numpy()
            per_plane_max = np.maximum(per_plane_max, plane_max)

print(f"\nWorst abs diff across ALL plies/games: {worst_overall:.6f}")
print(f"Num (game,ply) stacks with mismatch >1e-4: {n_plane_mismatch_total}")
print("Per-plane max diff (which channel disagrees):")
plane_names = (
    [f"own_{p}" for p in "PNBRQK"] + [f"opp_{p}" for p in "PNBRQK"]
    + ["castO_KS", "castO_QS", "castOpp_KS", "castOpp_QS", "ep", "turn",
       "movecount", "rep>=2", "rep>=3", "noprogress"]
)
for i, name in enumerate(plane_names):
    if per_plane_max[i] > 1e-4:
        print(f"  plane {i:2d} {name:12s}: max diff {per_plane_max[i]:.4f}")
if per_plane_max.max() <= 1e-4:
    print("  (no plane mismatches)")
