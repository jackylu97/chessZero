"""Quick head-to-head timing: GPU chess engine vs python-chess engine through
the actual self-play pipeline (BatchedMCTS + step + obs + legal_mask).

Runs one small batch of games per backend with a low max-ply cap so each
backend plays a comparable amount of work. Reports wall time, plies played,
and per-ply throughput.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import play_games_parallel, play_games_parallel_gpu


def build_network(cfg, game):
    cfg.hidden_planes = 32
    cfg.num_residual_blocks = 1
    cfg.fc_hidden = 32
    return MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size,
        hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks,
        latent_h=cfg.latent_h,
        latent_w=cfg.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
        use_consistency_loss=False,
        proj_hid=cfg.proj_hid,
        proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid,
        pred_out=cfg.pred_out,
        use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale,
        value_head_type=getattr(cfg, "value_head_type", "support"),
        draw_score=getattr(cfg, "draw_score", 0.0),
    )


def make_cfg(num_parallel: int, sims: int, max_plies: int):
    cfg = get_config("chess")
    cfg.num_parallel_games = num_parallel
    cfg.num_simulations = sims
    cfg.history_frames = 8
    cfg.sample_k = 50
    cfg.use_gumbel = False
    cfg.use_consistency_loss = False
    # Cap per-game plies so both backends finish in a similar horizon.
    cfg.use_gpu_chess = False  # toggled per-run below
    return cfg


def cap_max_plies(max_plies: int) -> None:
    """Patch both engines to terminate at `max_plies` plies for a fast bench."""
    ChessGame.max_plies = max_plies
    GpuChessGame.max_plies = max_plies


def time_run(label: str, fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    histories = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    plies = sum(len(h.actions) for h in histories)
    per_ply_ms = (dt / plies) * 1000 if plies else float("inf")
    games = len(histories)
    print(f"  {label:<14s}  {dt*1000:8.1f} ms total  "
          f"{plies:5d} plies in {games} games  "
          f"{per_ply_ms:6.2f} ms/ply  "
          f"{plies/dt:7.1f} plies/sec")
    return dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-games", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=10)
    p.add_argument("--sims", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"device={args.device}  num_games={args.num_games}  "
          f"max_plies={args.max_plies}  sims={args.sims}")
    print()

    cap_max_plies(args.max_plies)
    cfg = make_cfg(args.num_games, args.sims, args.max_plies)
    chess_game = ChessGame()
    network = build_network(cfg, chess_game).to(args.device).eval()

    # Warm both backends once so JIT/compile/cudnn warmups don't pollute.
    print("warmup:")
    torch.manual_seed(0)
    time_run("CPU warmup", lambda: play_games_parallel(
        network, chess_game, cfg, args.num_games, args.device, training_step=0,
    ))
    torch.manual_seed(0)
    time_run("GPU warmup", lambda: play_games_parallel_gpu(
        network, cfg, args.num_games, args.device, training_step=0,
    ))

    print()
    print("timed:")
    torch.manual_seed(1)
    cpu_t = time_run("python-chess", lambda: play_games_parallel(
        network, chess_game, cfg, args.num_games, args.device, training_step=0,
    ))
    torch.manual_seed(1)
    gpu_t = time_run("gpu chess",    lambda: play_games_parallel_gpu(
        network, cfg, args.num_games, args.device, training_step=0,
    ))

    print()
    speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
    print(f"speedup: gpu / cpu = {speedup:.2f}x")


if __name__ == "__main__":
    main()
