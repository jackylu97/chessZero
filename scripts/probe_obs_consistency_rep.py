"""Probe rep-plane consistency by FORCING repetitions.

Drives knight-shuffle repetition (Nf3-Ng1-Nf3...) on the GPU env and compares
the GPU's rep>=2 / rep>=3 planes (from the Zobrist ring) against the CPU
reconstruction's planes (from board.is_repetition). These are the planes that
the draw-collapsed buffer (~99% threefold) exercises hardest.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.chess import ChessGame, _move_to_action
import chess
from src.games.chess_gpu import GpuChessGame
from src.training.replay_buffer import GameHistory

N_FRAMES = 8
device = "cuda"
gpu = GpuChessGame()
gpu.max_plies = 300
cpu = ChessGame()
c = gpu.num_planes

# One game, single-game batch. Force a shuffle that repeats positions.
# White: Nb1-c3-b1-c3..., Black: Nb8-c6-b8-c6...
shuffle_uci = ["b1c3", "b8c6", "c3b1", "c6b8"]  # 4-ply cycle returns to start

state = gpu.reset_batch(1, device=device)
next_legal_mask = gpu.legal_mask(state)

board = chess.Board()  # parallel CPU board to derive action indices

obs_window = torch.zeros(1, N_FRAMES, c, 8, 8, device=device, dtype=torch.float32)
captured = []        # stacked obs seen
single_obs = []
actions = []

for ply in range(40):
    so = gpu.to_tensor_batch(state)
    obs_window = torch.roll(obs_window, shifts=1, dims=1)
    obs_window[:, 0] = so
    stacked = obs_window.reshape(1, N_FRAMES * c, 8, 8)

    uci = shuffle_uci[ply % len(shuffle_uci)]
    mv = chess.Move.from_uci(uci)
    a = _move_to_action(mv, board.turn)

    captured.append(stacked[0].cpu().clone())
    single_obs.append(so[0].cpu().clone())
    actions.append(a)

    board.push(mv)
    at = torch.tensor([a], dtype=torch.int64, device=device)
    state, rewards, done, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        print(f"Game ended at ply {ply} (done). "
              f"threefold={bool(state.terminal_threefold[0])}")
        break

L = len(actions)
print(f"Captured {L} plies of shuffle.")

# Inspect GPU rep planes directly (broadcast scalar -> take [0,0]).
print("\nGPU-seen rep planes (the stack newest frame, planes 19/20) per ply:")
for t in range(L):
    so = single_obs[t]
    print(f"  ply {t:2d} {shuffle_uci[t%4]}: rep>=2={float(so[19,0,0]):.0f} "
          f"rep>=3={float(so[20,0,0]):.0f} noprog={float(so[21,0,0]):.2f}")

# Reconstruct via training path.
gh = GameHistory(game_name="chess")
for t in range(L):
    gh.observations.append(single_obs[t])
    gh.actions.append(int(actions[t]))
    p = np.zeros(cpu.action_space_size, dtype=np.float32); p[int(actions[t])] = 1.0
    gh.policies.append(p)
    gh.root_values.append(0.0)
    gh.rewards.append(0.0)
gh.game_outcome = 0
gh.observations.append(single_obs[-1])

compact = gh.to_compact_dict()
recon = GameHistory.from_compact_dict(compact, ChessGame())

print("\nCPU-reconstructed rep planes (single-frame to_tensor) per ply:")
for t in range(L):
    o = recon.observations[t]
    print(f"  ply {t:2d} {shuffle_uci[t%4]}: rep>=2={float(o[19,0,0]):.0f} "
          f"rep>=3={float(o[20,0,0]):.0f} noprog={float(o[21,0,0]):.2f}")

# Compare full stacks.
print("\nStack comparison (net-SAW vs net-TRAINS):")
worst = 0.0
per_plane = np.zeros(c)
mismatch_plies = []
for t in range(L):
    saw = captured[t]
    trained = recon._stack_history(t, N_FRAMES)
    diff = (saw - trained).abs()
    m = float(diff.max())
    worst = max(worst, m)
    if m > 1e-4:
        mismatch_plies.append(t)
        pf = diff.view(N_FRAMES, c, 8, 8).amax(dim=(0, 2, 3)).numpy()
        per_plane = np.maximum(per_plane, pf)

print(f"Worst abs diff: {worst:.4f}")
print(f"Mismatch plies: {mismatch_plies}")
plane_names = (
    [f"own_{p}" for p in "PNBRQK"] + [f"opp_{p}" for p in "PNBRQK"]
    + ["castO_KS","castO_QS","castOpp_KS","castOpp_QS","ep","turn",
       "movecount","rep>=2","rep>=3","noprogress"]
)
for i,name in enumerate(plane_names):
    if per_plane[i] > 1e-4:
        print(f"  plane {i:2d} {name:12s}: max diff {per_plane[i]:.4f}")
