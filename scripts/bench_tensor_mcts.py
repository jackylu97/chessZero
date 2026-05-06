"""Head-to-head bench: play_games_parallel_gpu + BatchedMCTS vs TensorMCTS.

Both backends share the same GPU-resident chess env (`GpuChessGame`); only
the MCTS implementation swaps between them via ``cfg.use_tensor_mcts``.

The new path's main benefit is sync collapse:

- BatchedMCTS:  3 GPU→CPU transfers per simulation × num_simulations
                × plies (rewards.tolist + values.tolist + probs.cpu).
- TensorMCTS:   1 GPU→CPU transfer per ply (compat shim copies root tensors
                back at end of run_batch). Selection has a per-step
                early-exit `bool(still_walking.any())` sync but it scales
                with average path depth, not with sims × 3.

Runs each backend twice (warmup + timed) and reports wall time, plies, and
per-ply / per-sim throughput. The GPU env (`GpuChessGame`) is held constant.

Usage:
    python scripts/bench_tensor_mcts.py --num-games 64 --sims 32 --max-plies 20
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
from src.training.self_play import (
    play_games_parallel_gpu,
    play_games_parallel_gpu_resident,
)


def build_network(cfg, game):
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


def make_cfg(num_parallel: int, sims: int, hidden_planes: int,
             num_blocks: int, hidden_dtype: str) -> object:
    cfg = get_config("chess")
    cfg.num_parallel_games = num_parallel
    cfg.num_simulations = sims
    cfg.history_frames = 8
    cfg.sample_k = 50
    cfg.use_gumbel = False
    cfg.use_consistency_loss = False
    cfg.hidden_planes = hidden_planes
    cfg.num_residual_blocks = num_blocks
    cfg.fc_hidden = hidden_planes
    cfg.tensor_mcts_hidden_dtype = hidden_dtype
    cfg.use_gpu_chess = True  # ensures play_games_parallel_gpu path
    return cfg


def cap_max_plies(max_plies: int) -> None:
    ChessGame.max_plies = max_plies
    GpuChessGame.max_plies = max_plies


def time_run(label: str, fn, device: str) -> tuple[float, int, int]:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    histories = fn()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    plies = sum(len(h.actions) for h in histories)
    games = len(histories)
    per_ply_ms = (dt / plies) * 1000 if plies else float("inf")
    print(
        f"  {label:<26s}  {dt*1000:9.1f} ms total  "
        f"{plies:5d} plies in {games} games  "
        f"{per_ply_ms:7.2f} ms/ply  "
        f"{plies/dt:8.1f} plies/sec"
    )
    return dt, plies, games


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-games", type=int, default=64)
    p.add_argument("--max-plies", type=int, default=20)
    p.add_argument("--sims", type=int, default=32)
    p.add_argument("--hidden-planes", type=int, default=64,
                   help="Trim from preset 256 so MCTS overhead is visible "
                        "next to the network forward.")
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--hidden-dtype", default="float16",
                   help="TensorMCTS storage dtype: float32 / float16 / bfloat16.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--repeat", type=int, default=3,
                   help="Timed runs per backend (median reported).")
    args = p.parse_args()

    print(
        f"device={args.device}  num_games={args.num_games}  sims={args.sims}  "
        f"max_plies={args.max_plies}  hidden_planes={args.hidden_planes}  "
        f"num_blocks={args.num_blocks}  hidden_dtype={args.hidden_dtype}  "
        f"repeat={args.repeat}"
    )
    print()

    cap_max_plies(args.max_plies)
    cfg = make_cfg(
        args.num_games, args.sims, args.hidden_planes, args.num_blocks,
        args.hidden_dtype,
    )
    chess_game = ChessGame()
    network = build_network(cfg, chess_game).to(args.device).eval()

    def run_batched():
        cfg.use_tensor_mcts = False
        return play_games_parallel_gpu(
            network, cfg, args.num_games, args.device, training_step=0,
        )

    def run_tensor():
        cfg.use_tensor_mcts = True
        cfg.tensor_mcts_select_backend = "compile"
        return play_games_parallel_gpu(
            network, cfg, args.num_games, args.device, training_step=0,
        )

    def run_tensor_resident():
        cfg.use_tensor_mcts = True
        cfg.tensor_mcts_select_backend = "compile"
        return play_games_parallel_gpu_resident(
            network, cfg, args.num_games, args.device, training_step=0,
        )

    def run_tensor_triton():
        cfg.use_tensor_mcts = True
        cfg.tensor_mcts_select_backend = "triton"
        return play_games_parallel_gpu_resident(
            network, cfg, args.num_games, args.device, training_step=0,
        )

    # Warm all paths — first call always pays cudnn/cublas/multinomial init costs.
    print("warmup:")
    torch.manual_seed(0); time_run("BatchedMCTS warmup", run_batched, args.device)
    torch.manual_seed(0); time_run("TensorMCTS warmup",  run_tensor,  args.device)
    torch.manual_seed(0); time_run("TensorMCTS+GPU-resident warmup", run_tensor_resident, args.device)
    torch.manual_seed(0); time_run("TensorMCTS+Triton warmup", run_tensor_triton, args.device)

    print()
    print(f"timed (median over {args.repeat} runs):")

    def repeat(label, fn):
        # Pick the FASTEST run rather than median — recompiles / first-touch
        # init costs occasionally inflate one run, and we want the achievable
        # steady-state, not the average over outliers.
        results = []
        for i in range(args.repeat):
            torch.manual_seed(100 + i)
            dt, plies, _ = time_run(f"{label} run {i+1}", fn, args.device)
            results.append((dt, plies))
        results.sort(key=lambda x: x[0])
        best_dt, best_plies = results[0]
        return best_dt, best_plies

    bmcts_dt, bmcts_plies = repeat("BatchedMCTS", run_batched)
    tmcts_dt, tmcts_plies = repeat("TensorMCTS",  run_tensor)
    res_dt, res_plies = repeat("TensorMCTS+GPU-resident", run_tensor_resident)
    tri_dt, tri_plies = repeat("TensorMCTS+Triton", run_tensor_triton)

    print()
    bmcts_pps = bmcts_plies / bmcts_dt
    tmcts_pps = tmcts_plies / tmcts_dt
    res_pps = res_plies / res_dt
    tri_pps = tri_plies / tri_dt
    print(
        f"summary: BatchedMCTS                   {bmcts_pps:7.1f} plies/s  (1.00× baseline)\n"
        f"         TensorMCTS (compile)          {tmcts_pps:7.1f} plies/s  ({tmcts_pps/bmcts_pps:.2f}×)\n"
        f"         TensorMCTS+GPU-resident       {res_pps:7.1f} plies/s  ({res_pps/bmcts_pps:.2f}×)\n"
        f"         TensorMCTS+Triton+GPU-res     {tri_pps:7.1f} plies/s  ({tri_pps/bmcts_pps:.2f}×)"
    )

    # Approximate sync counts (assumes one batch of N games, each with `plies`
    # plies and `sims` simulations per ply).
    bmcts_syncs = 3 * args.sims * bmcts_plies / bmcts_plies  # per-ply: 3·sims
    print()
    print(f"approx syncs/ply: BatchedMCTS = {3*args.sims} (3·sims for rewards/values/probs)")
    print(f"                  TensorMCTS  ≈ avg_depth + 1 (selection any-walk + compat shim)")


if __name__ == "__main__":
    main()
