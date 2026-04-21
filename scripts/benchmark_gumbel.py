"""Microbench: Sampled MuZero vs Plain Gumbel MuZero self-play throughput.

Times ``BatchedMCTS.run_batch`` at chess config with N parallel games under:
  (a) Sampled MuZero (Hubert 2021 Proposed Modification): use_gumbel=False, sample_k.
  (b) Plain Gumbel MuZero (Danihelka 2022): use_gumbel=True, gumbel_num_considered,
      plus sample_k for leaf expansion.

Uses a random-init network (we're measuring compute throughput, not policy
quality). Runs a warmup batch, then K timed trials per config. Reports mean
seconds per run_batch call, games/sec, and the Gumbel/Sampled ratio.

Plan target (plan_gumbel_muzero.md G4): within 1.3× wall-clock of Sampled.

Usage:
    .venv/bin/python scripts/benchmark_gumbel.py --n 128 --sims 100 --trials 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.mcts.mcts import BatchedMCTS
from src.model.muzero_net import MuZeroNetwork


def make_chess_net(config, device: str) -> MuZeroNetwork:
    game = ChessGame()
    net = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h, latent_w=config.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
    )
    net.to(device)
    net.eval()
    return net


def time_run_batch(net, game, config, n: int, trials: int, device: str) -> tuple[float, float]:
    """Return (mean_seconds_per_batch, sims_per_sec)."""
    mcts = BatchedMCTS(net, game, config, device)
    state = game.reset()
    obs_list = [game.to_tensor(state) for _ in range(n)]
    legal_list = [game.legal_actions(state) for _ in range(n)]

    # Warmup — CUDA init, allocator caching, first-call JIT compile.
    _ = mcts.run_batch(obs_list, legal_list, add_noise=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = mcts.run_batch(obs_list, legal_list, add_noise=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_sec = sum(times) / len(times)
    sims_per_sec = (n * config.num_simulations) / mean_sec
    return mean_sec, sims_per_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="parallel games per run_batch call")
    ap.add_argument("--sims", type=int, default=100, help="MCTS simulations per root")
    ap.add_argument("--trials", type=int, default=5, help="timed run_batch calls per config")
    ap.add_argument("--gumbel-m", type=int, default=50, help="gumbel_num_considered")
    ap.add_argument("--sample-k", type=int, default=50, help="sample_k (leaf) — used by both")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print(f"Config: N={args.n}, sims={args.sims}, trials={args.trials}, "
          f"gumbel_m={args.gumbel_m}, sample_k={args.sample_k}")
    print()

    game = ChessGame()
    base = get_config("chess")
    base.num_simulations = args.sims
    base.sample_k = args.sample_k
    base.num_parallel_games = args.n

    net = make_chess_net(base, args.device)
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Network: {params_m:.2f}M params")
    print()

    # --- Sampled MuZero ----
    cfg_sampled = get_config("chess")
    cfg_sampled.num_simulations = args.sims
    cfg_sampled.sample_k = args.sample_k
    cfg_sampled.num_parallel_games = args.n
    cfg_sampled.use_gumbel = False

    sampled_sec, sampled_sps = time_run_batch(
        net, game, cfg_sampled, args.n, args.trials, args.device,
    )
    print(f"Sampled MuZero: {sampled_sec*1000:.1f} ms/batch  "
          f"({sampled_sps:,.0f} sims/s, {args.n/sampled_sec:,.1f} roots/s)")

    # --- Plain Gumbel MuZero ----
    cfg_gumbel = get_config("chess")
    cfg_gumbel.num_simulations = args.sims
    cfg_gumbel.sample_k = args.sample_k
    cfg_gumbel.num_parallel_games = args.n
    cfg_gumbel.use_gumbel = True
    cfg_gumbel.gumbel_num_considered = args.gumbel_m
    cfg_gumbel.use_gumbel_noise = True

    gumbel_sec, gumbel_sps = time_run_batch(
        net, game, cfg_gumbel, args.n, args.trials, args.device,
    )
    print(f"Gumbel MuZero:  {gumbel_sec*1000:.1f} ms/batch  "
          f"({gumbel_sps:,.0f} sims/s, {args.n/gumbel_sec:,.1f} roots/s)")

    print()
    ratio = gumbel_sec / sampled_sec
    print(f"Ratio (gumbel / sampled): {ratio:.3f}x  "
          f"[plan target: <= 1.30x]")


if __name__ == "__main__":
    main()
