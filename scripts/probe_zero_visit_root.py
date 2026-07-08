"""Probe: can run_batch_gpu ever produce a root whose summed child visits are 0
(the all-zero-visits case that makes select_action_gpu's multinomial crash on the
sampled path)? Also: is the sampled path ALWAYS taken (temp_final != 0)?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import chess
import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.games import chess_gpu as cg
from src.mcts.tensor_mcts import TensorMCTS, select_action_gpu
from scripts.eval_checkpoint_health import build_network


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config("chess_small")
    cfg.device = device
    print(f"temperature_init={cfg.temperature_init} temperature_final={cfg.temperature_final} "
          f"temperature_drop_step={cfg.temperature_drop_step} num_simulations={cfg.num_simulations}")
    print(f"  -> sampled path (multinomial) taken whenever temp != 0; "
          f"temp_final={cfg.temperature_final} => greedy NEVER used in gpu-resident self-play\n")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    game = ChessGame()
    net = build_network(ckpt, game, cfg, device)
    gpu = cg.GpuChessGame()

    # Build a batch of varied positions: openings, midgame, and near-terminal
    # (very few legal moves) positions to stress the root.
    rng = random.Random(0)
    boards = []
    for _ in range(120):
        b = chess.Board()
        for _ in range(rng.randint(0, 80)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(rng.choice(ms))
        if not b.is_game_over() and any(True for _ in b.legal_moves):
            boards.append(b)
    # Add hand-built positions with exactly ONE legal move (king in check, one escape).
    boards.append(chess.Board("7k/8/8/8/8/8/5Q2/6RK b - - 0 1"))  # black king cornered
    boards.append(chess.Board("k7/8/1K6/8/8/8/8/1Q6 b - - 0 1"))
    print(f"built {len(boards)} positions")

    mcts = TensorMCTS(net, game, cfg, device=device,
                      hidden_dtype=torch.float16, amp_dtype=torch.float16,
                      select_backend=getattr(cfg, "tensor_mcts_select_backend", "compile"))

    state = gpu.from_python_chess(boards, device=device)
    obs = gpu.to_tensor_batch(state)
    legal_mask = gpu.legal_mask(state)
    # sentinel for any all-zero (terminal) rows, mirroring self_play loop
    any_legal = legal_mask.any(dim=1, keepdim=True)
    sentinel = torch.zeros_like(legal_mask); sentinel[:, 0] = True
    legal_mask = torch.where(any_legal, legal_mask, sentinel)

    # history_frames stacking: replicate single frame n_frames times is wrong;
    # just zero-pad older frames (matches first-ply behavior).
    n_frames = int(cfg.history_frames)
    c, h, w = obs.shape[1:]
    if n_frames > 1:
        stacked = torch.zeros(obs.shape[0], n_frames * c, h, w, device=device, dtype=obs.dtype)
        stacked[:, :c] = obs
    else:
        stacked = obs

    root = mcts.run_batch_gpu(stacked, legal_mask, add_noise=True)
    ca = root["child_actions"]; cv = root["child_visits"]
    # summed visits per unique action (dedup via scatter)
    A = game.action_space_size
    valid = ca != -1
    visits_f = torch.where(valid, cv.float(), torch.zeros_like(cv.float()))
    dense = torch.zeros(ca.shape[0], A, device=device)
    dense.scatter_add_(1, ca.clamp(min=0).long(), visits_f)
    row_sums = dense.sum(dim=1)
    n_zero = int((row_sums <= 0).sum().item())
    print(f"\nroots with ALL-ZERO summed child visits: {n_zero} / {ca.shape[0]}")
    print(f"min row visit sum = {row_sums.min().item():.1f}, "
          f"max = {row_sums.max().item():.1f}, mean = {row_sums.mean().item():.1f}")

    # Now actually call select_action_gpu on the SAMPLED path (temp=temp_final).
    temp = torch.full((ca.shape[0],), float(cfg.temperature_final), device=device)
    try:
        action, policy = select_action_gpu(ca, cv, temp, A)
        print(f"select_action_gpu (sampled path, T={cfg.temperature_final}) OK; "
              f"actions finite={bool(torch.isfinite(action.float()).all())}")
    except Exception as e:
        print(f"select_action_gpu CRASHED on sampled path: {type(e).__name__}: {e}")

    # Force the pathological all-zero case directly to confirm the latent crash.
    print("\n--- forced all-zero-visit root (synthetic) on sampled path ---")
    ca_z = torch.tensor([[3, 7, -1, -1]], dtype=torch.int32, device=device)
    cv_z = torch.zeros(1, 4, dtype=torch.int32, device=device)
    try:
        a, p = select_action_gpu(ca_z, cv_z, torch.tensor(0.1, device=device), A)
        print("  no crash; action=", int(a.item()))
    except Exception as e:
        print(f"  CRASH (confirms latent bug): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
