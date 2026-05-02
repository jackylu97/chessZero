"""Benchmark GpuChessGame (chess_gpu) at varying N, on CPU + CUDA, with/without
torch.compile. Compares against python-chess as the per-game reference.

Used to validate Phase 4 of the GPU chess plan: only worth pursuing if
compile + CUDA residency beats CPU torch by a wide margin and crosses
parity with python-chess at meaningful self-play scale.
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

from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame


def random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        plies = rng.randint(0, 60)
        for _ in range(plies):
            if b.is_game_over(claim_draw=False):
                break
            b.push(rng.choice(list(b.legal_moves)))
        out.append(b.copy())
    return out


@contextmanager
def timer(name: str, sync_cuda: bool = False):
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {name:<40s} {dt*1000:>10.2f} ms")


def bench_python_chess(boards: list[chess.Board]) -> float:
    """Per-game `legal_actions` via python-chess (oracle baseline)."""
    cg = ChessGame()
    from src.games.base import GameState
    states = [GameState(board=b, current_player=1 if b.turn else -1) for b in boards]
    t0 = time.perf_counter()
    total = 0
    for s in states:
        legals = cg.legal_actions(s)
        total += len(legals)
    return time.perf_counter() - t0


def bench_legal_mask(gg: GpuChessGame, state, sync: bool, n_iters: int = 10) -> float:
    # Warmup.
    for _ in range(3):
        _ = gg.legal_mask(state)
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = gg.legal_mask(state)
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters


def bench_step_batch(gg: GpuChessGame, state, actions, sync: bool, n_iters: int = 10) -> float:
    for _ in range(3):
        _ = gg.step_batch(state, actions)
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = gg.step_batch(state, actions)
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256, help="Batch size.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--compile", action="store_true", help="Apply torch.compile to legal_mask/step_batch.")
    args = p.parse_args()

    sync = args.device == "cuda"
    print(f"=== chess_gpu bench: device={args.device}, N={args.n}, iters={args.iters}, compile={args.compile} ===\n")

    boards = random_positions(args.n, seed=1)

    # python-chess baseline (CPU only).
    pc_total = bench_python_chess(boards)
    print(f"python-chess legal_actions over {args.n} games: {pc_total*1000:.2f} ms total "
          f"({pc_total/args.n*1e6:.1f} µs/game)\n")

    # Build batched state on the chosen device.
    gg = GpuChessGame()
    state = gg.from_python_chess(boards, device=args.device)

    # Random legal action per game (used for step bench).
    cg = ChessGame()
    from src.games.base import GameState
    actions_list = []
    for b in boards:
        st = GameState(board=b, current_player=1 if b.turn else -1)
        legals = cg.legal_actions(st)
        actions_list.append(legals[0] if legals else 0)
    actions = torch.tensor(actions_list, dtype=torch.int64, device=args.device)

    if args.compile:
        gg.legal_mask = torch.compile(gg.legal_mask, dynamic=False, fullgraph=False)
        gg.step_batch = torch.compile(gg.step_batch, dynamic=False, fullgraph=False)
        print("(compile=True; first call below includes compilation)\n")

    print(f"GpuChessGame on {args.device}:")
    t_legal = bench_legal_mask(gg, state, sync=sync, n_iters=args.iters)
    print(f"  legal_mask                              {t_legal*1000:>10.2f} ms/call ({t_legal/args.n*1e6:.2f} µs/game)")
    t_step = bench_step_batch(gg, state, actions, sync=sync, n_iters=args.iters)
    print(f"  step_batch                              {t_step*1000:>10.2f} ms/call ({t_step/args.n*1e6:.2f} µs/game)")

    print(f"\nspeedup vs python-chess: legal_mask {pc_total/(t_legal):.1f}x  step ~ N/A (different op)")


if __name__ == "__main__":
    main()
