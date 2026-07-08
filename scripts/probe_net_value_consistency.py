"""End-to-end: feed the EXACT stacked obs the net saw in self-play AND the
buffer-reconstructed stacked obs into the REAL checkpoint network. Assert the
value/policy outputs are identical. Closes the loop: if obs match bit-for-bit
the net outputs must match; this catches any latent dtype/layout issue the raw
tensor compare could miss.

CPU, map_location='cpu' (safe alongside live GPU training).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.games.chess import ChessGame, _move_to_action
import chess
from src.games.chess_gpu import GpuChessGame
from src.training.replay_buffer import GameHistory
from scripts.eval_checkpoint_health import build_network

CKPT = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.pt"
N_FRAMES = 8
gpu_device = "cuda"

cfg = get_config("chess_small"); cfg.device = "cpu"
game = ChessGame()
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
net = build_network(ckpt, game, cfg, "cpu")
net.eval()

# Drive a forced-shuffle (repetition) game on GPU, capturing what the net saw.
gpu = GpuChessGame(); gpu.max_plies = 60
c = gpu.num_planes
state = gpu.reset_batch(1, device=gpu_device)
next_legal_mask = gpu.legal_mask(state)
obs_window = torch.zeros(1, N_FRAMES, c, 8, 8, device=gpu_device, dtype=torch.float32)

shuffle = ["e2e4","e7e5","g1f3","b8c6","f3g1","c6b8","g1f3","b8c6","f3g1","c6b8"]
board = chess.Board()
captured, single_obs, actions = [], [], []
for ply in range(len(shuffle)):
    so = gpu.to_tensor_batch(state)
    obs_window = torch.roll(obs_window, shifts=1, dims=1)
    obs_window[:, 0] = so
    stacked = obs_window.reshape(1, N_FRAMES * c, 8, 8)
    mv = chess.Move.from_uci(shuffle[ply]); a = _move_to_action(mv, board.turn)
    captured.append(stacked[0].cpu().clone()); single_obs.append(so[0].cpu().clone())
    actions.append(a); board.push(mv)
    at = torch.tensor([a], dtype=torch.int64, device=gpu_device)
    state, _, _, next_legal_mask = gpu.step_batch_with_legal(state, at)
    if bool(state.done):
        break

L = len(actions)
gh = GameHistory(game_name="chess")
for t in range(L):
    gh.observations.append(single_obs[t]); gh.actions.append(int(actions[t]))
    p = np.zeros(game.action_space_size, dtype=np.float32); p[int(actions[t])] = 1.0
    gh.policies.append(p); gh.root_values.append(0.0); gh.rewards.append(0.0)
gh.game_outcome = 0
gh.observations.append(single_obs[-1])
recon = GameHistory.from_compact_dict(gh.to_compact_dict(), ChessGame())

print(f"L={L}. Comparing net outputs on self-play-seen vs trained-reconstructed stacks:")
max_v_diff = 0.0
for t in range(L):
    saw = captured[t].unsqueeze(0)
    trn = recon._stack_history(t, N_FRAMES).unsqueeze(0)
    with torch.no_grad():
        _, pl_s, v_s = net.initial_inference(saw)
        _, pl_t, v_t = net.initial_inference(trn)
    # value head is WDL (3-vec); convert to scalar W-L.
    def scalar(v):
        v = torch.softmax(v, dim=-1) if v.shape[-1] == 3 else v
        if v.shape[-1] == 3:
            return float(v[0, 0] - v[0, 2])
        return float(v.reshape(-1)[0])
    vs, vt = scalar(v_s), scalar(v_t)
    vd = abs(vs - vt)
    pd = float((torch.softmax(pl_s, -1) - torch.softmax(pl_t, -1)).abs().max())
    max_v_diff = max(max_v_diff, vd)
    # Per-plane raw-tensor diff to localize WHICH channel diverges.
    pdiff = (saw - trn).abs().view(N_FRAMES, c, 8, 8).amax(dim=(0, 2, 3)).numpy()
    bad = [(i, float(pdiff[i])) for i in range(c) if pdiff[i] > 1e-4]
    print(f"  ply {t:2d} {shuffle[t]}: v_saw={vs:+.5f} v_trained={vt:+.5f} "
          f"|Δv|={vd:.2e} |Δpolicy|max={pd:.2e}  planes_diff={bad}")

print(f"\nMax |Δvalue| over game: {max_v_diff:.2e}")
print("PASS (net sees identical input)" if max_v_diff < 1e-4
      else "FAIL (net trains on different input than it played!)")
