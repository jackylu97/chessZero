"""Probe: does the training-time re-stacked observation EXACTLY equal the
stacked observation the net searched on during GPU-resident self-play?

The self-play loop stores single-frame obs per ply but searches on an 8-frame
rolling window. The replay buffer reconstructs the 8-frame stack at sample time
from the stored single frames. If the reconstruction differs from what was
searched, the policy/value target is paired with the WRONG observation -> the
net learns a target generated under a different input -> silent corruption.

We re-run the GPU-resident loop while CAPTURING the exact stacked_obs the net saw
each ply, then compare against stack_with_history applied to the stored history.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network
import src.training.self_play as sp


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config("chess_small")
    cfg.device = device
    cfg.use_gpu_chess = True
    cfg.use_tensor_mcts = True
    cfg.use_gpu_resident_self_play = True
    cfg.max_plies = 30
    n_games = 6
    n_frames = int(cfg.history_frames)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    game = ChessGame()
    net = build_network(ckpt, game, cfg, device)

    # Monkeypatch run_batch_gpu to capture the stacked_obs it receives per ply.
    captured = {"obs": []}  # list over plies of [N, C*F, H, W] cpu tensors
    from src.mcts.tensor_mcts import TensorMCTS
    orig = TensorMCTS.run_batch_gpu

    def patched(self, obs_batch, legal_mask, add_noise=True):
        captured["obs"].append(obs_batch.detach().float().cpu().numpy().copy())
        return orig(self, obs_batch, legal_mask, add_noise=add_noise)

    TensorMCTS.run_batch_gpu = patched
    try:
        torch.manual_seed(0)
        histories = sp.play_games_parallel_gpu_resident(
            net, cfg, n_games, device=device, training_step=0
        )
    finally:
        TensorMCTS.run_batch_gpu = orig

    # captured["obs"][ply] is [N, C*F, H, W]; game g's searched stack at ply t is
    # captured["obs"][t][g]. The stored history single-frame obs reconstruction:
    n_random = int(getattr(cfg, "random_opening_plies", 0))
    max_diff = 0.0
    n_checked = 0
    mism = 0
    for g, h in enumerate(histories):
        single_frames = [o.float().numpy() for o in h.observations[:-1]]  # drop terminal
        for t in range(len(h.actions)):
            ply = t  # search ran every ply >= n_random; for ply < n_random no search
            if ply < n_random:
                continue
            if ply >= len(captured["obs"]):
                continue
            searched = captured["obs"][ply][g]  # [C*F, H, W]
            # Reconstruct as the buffer would at training time:
            cur = torch.from_numpy(single_frames[t])
            prior = [torch.from_numpy(single_frames[j]) for j in range(t)]
            restacked = stack_with_history(cur, prior, n_frames).numpy()
            d = np.abs(searched - restacked).max()
            max_diff = max(max_diff, float(d))
            n_checked += 1
            if d > 1e-4:
                mism += 1
                if mism <= 5:
                    print(f"  MISMATCH g={g} ply={t} maxdiff={d:.4f}")

    print(f"\nchecked {n_checked} (game,ply) stacks across {len(histories)} games, "
          f"n_frames={n_frames}")
    print(f"searched-stack vs training-restack: {mism} mismatches, max|diff|={max_diff:.2e}")
    if mism == 0:
        print("OK: training reconstruction is identical to the searched observation.")


if __name__ == "__main__":
    main()
