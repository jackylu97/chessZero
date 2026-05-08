"""Head-to-head bench: ChessGame (per-game python-chess loop) vs GpuChessGame
(batched torch). Reports per-call wall and per-game throughput across the
three env primitives — to_tensor, legal_actions/legal_mask, step — at
varying batch sizes, on CPU eager / CUDA eager / CUDA compiled.

The same random corpus is used for all backends so the comparison is fair.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.games.base import GameState
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame


def random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    """Generate n non-terminal positions (so step always has legal moves)."""
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        plies = rng.randint(0, 60)
        for _ in range(plies):
            if b.is_game_over(claim_draw=False):
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over(claim_draw=False) and not b.is_repetition(3):
            out.append(b.copy())
    return out


def time_call(fn, sync: bool, iters: int) -> float:
    """Avg per-call wall time over `iters` repetitions, after a 3-call warmup."""
    for _ in range(3):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_at_n(n: int, device: str, compile_: bool, iters: int) -> dict:
    """Bench all three primitives at batch size `n`. Returns timings dict."""
    sync = device == "cuda"
    boards = random_positions(n, seed=1)
    cg = ChessGame()
    cpu_states = [GameState(board=b.copy(), current_player=1 if b.turn else -1) for b in boards]
    cpu_actions = [
        cg.legal_actions(s)[0] if cg.legal_actions(s) else 0 for s in cpu_states
    ]

    # ---- CPU per-game (ChessGame) ----
    def cpu_to_tensor():
        for s in cpu_states:
            _ = cg.to_tensor(s)

    def cpu_legal():
        for s in cpu_states:
            _ = cg.legal_actions(s)

    def cpu_step():
        # `step` mutates by copying internally; pass fresh-copy states each call
        # by re-cloning. Cloning cost included — matches what self-play does.
        local = [GameState(board=s.board.copy(), current_player=s.current_player) for s in cpu_states]
        for s, a in zip(local, cpu_actions):
            _, _, _ = cg.step(s, a)

    cpu_t_to_tensor = time_call(cpu_to_tensor, sync=False, iters=iters)
    cpu_t_legal = time_call(cpu_legal, sync=False, iters=iters)
    cpu_t_step = time_call(cpu_step, sync=False, iters=iters)

    # ---- GPU batched (GpuChessGame) ----
    gg = GpuChessGame()
    state = gg.from_python_chess(boards, device=device)
    actions_t = torch.tensor(cpu_actions, dtype=torch.int64, device=device)

    if compile_:
        gg_to_tensor = torch.compile(gg.to_tensor_batch, dynamic=False, fullgraph=False)
        gg_legal = torch.compile(gg.legal_mask, dynamic=False, fullgraph=False)
        gg_step = torch.compile(gg.step_batch, dynamic=False, fullgraph=False)
    else:
        gg_to_tensor = gg.to_tensor_batch
        gg_legal = gg.legal_mask
        gg_step = gg.step_batch

    def gpu_to_tensor():
        _ = gg_to_tensor(state)

    def gpu_legal():
        _ = gg_legal(state)

    def gpu_step():
        _ = gg_step(state, actions_t)

    gpu_t_to_tensor = time_call(gpu_to_tensor, sync=sync, iters=iters)
    gpu_t_legal = time_call(gpu_legal, sync=sync, iters=iters)
    gpu_t_step = time_call(gpu_step, sync=sync, iters=iters)

    return {
        "n": n,
        "device": device,
        "compile": compile_,
        "cpu": {"to_tensor": cpu_t_to_tensor, "legal": cpu_t_legal, "step": cpu_t_step},
        "gpu": {"to_tensor": gpu_t_to_tensor, "legal": gpu_t_legal, "step": gpu_t_step},
    }


def fmt_row(label: str, n: int, op: str, cpu: float, gpu: float) -> str:
    speedup = cpu / gpu if gpu > 0 else float("inf")
    cpu_ms = cpu * 1000
    gpu_ms = gpu * 1000
    cpu_per_game = cpu / n * 1e6
    gpu_per_game = gpu / n * 1e6
    return (
        f"{label:<22s} N={n:<5d} {op:<11s} "
        f"CPU {cpu_ms:>8.2f} ms ({cpu_per_game:>6.1f} µs/game)  "
        f"GPU {gpu_ms:>8.2f} ms ({gpu_per_game:>6.1f} µs/game)  "
        f"speedup {speedup:>5.1f}×"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="64,256,1024,4096", help="Comma-separated batch sizes.")
    p.add_argument("--devices", default="cpu,cuda", help="Comma-separated devices to bench.")
    p.add_argument("--no-compile", action="store_true", help="Skip the compiled CUDA pass.")
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    devices = [d for d in args.devices.split(",") if d == "cpu" or torch.cuda.is_available()]

    print("=" * 110)
    print("ChessGame (single-state Python loop)  vs  GpuChessGame (batched torch)")
    print("=" * 110)

    for device in devices:
        # Eager.
        for n in ns:
            r = bench_at_n(n, device, compile_=False, iters=args.iters)
            label = f"{device} eager"
            print(fmt_row(label, n, "to_tensor", r["cpu"]["to_tensor"], r["gpu"]["to_tensor"]))
            print(fmt_row(label, n, "legal", r["cpu"]["legal"], r["gpu"]["legal"]))
            print(fmt_row(label, n, "step", r["cpu"]["step"], r["gpu"]["step"]))
            print()

        # Compiled (CUDA only — compile on CPU is heavy and rarely used).
        if device == "cuda" and not args.no_compile:
            print("--- compiled ---")
            for n in ns:
                r = bench_at_n(n, device, compile_=True, iters=args.iters)
                label = f"{device} compiled"
                print(fmt_row(label, n, "to_tensor", r["cpu"]["to_tensor"], r["gpu"]["to_tensor"]))
                print(fmt_row(label, n, "legal", r["cpu"]["legal"], r["gpu"]["legal"]))
                print(fmt_row(label, n, "step", r["cpu"]["step"], r["gpu"]["step"]))
                print()


if __name__ == "__main__":
    main()
