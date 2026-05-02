"""Benchmark per-move wall-time breakdown of batched self-play.

Mirrors `play_games_parallel` but instruments each call:
  env.to_tensor / env.legal_actions / env.stack_history / env.step
  mcts.run_batch (inclusive)
  net.initial_inference / net.recurrent_inference (subsets of run_batch)
  mcts.select_action

Network inference is wrapped by monkey-patching the bound methods so the
timing covers the same `recurrent_inference` calls MCTS makes per simulation.

Used to validate the GPU-resident chess engine plan: only worth pursuing if
root-side env ops are a non-trivial fraction of move time.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.games.chess import ChessGame
from src.mcts.mcts import BatchedMCTS, select_action
from src.model.muzero_net import MuZeroNetwork
from src.training.replay_buffer import GameHistory, stack_with_history


class Buckets:
    def __init__(self, sync_cuda: bool):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self.sync_cuda = sync_cuda

    @contextmanager
    def timed(self, name: str):
        if self.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                torch.cuda.synchronize()
            self.t[name] += time.perf_counter() - t0
            self.n[name] += 1

    def add(self, name: str, dt: float, count: int = 1):
        self.t[name] += dt
        self.n[name] += count

    def report(self, total: float, moves: int, n_games: int):
        rows = sorted(self.t.keys(), key=lambda k: -self.t[k])
        net_total = self.t.get("net.initial_inference", 0.0) + self.t.get(
            "net.recurrent_inference", 0.0
        )
        run_batch_total = self.t.get("mcts.run_batch", 0.0)
        mcts_internal = max(0.0, run_batch_total - net_total)

        print(f"\n{'bucket':<30}{'total_s':>10}{'count':>10}{'us/call':>12}{'ms/move':>10}{'%':>8}")
        print("-" * 80)
        for k in rows:
            ts = self.t[k]
            n = self.n[k]
            us = (ts / n) * 1e6 if n else 0.0
            ms_per_move = (ts / moves) * 1000 if moves else 0.0
            pct = (ts / total) * 100 if total else 0.0
            print(f"{k:<30}{ts:>10.3f}{n:>10}{us:>12.1f}{ms_per_move:>10.1f}{pct:>7.1f}%")
        print("-" * 80)
        if run_batch_total > 0:
            pct_int = mcts_internal / total * 100
            print(
                f"{'(derived) mcts.internal_only':<30}"
                f"{mcts_internal:>10.3f}{'-':>10}{'-':>12}"
                f"{(mcts_internal / moves) * 1000:>10.1f}"
                f"{pct_int:>7.1f}%"
            )

        env_keys = [k for k in rows if k.startswith("env.")]
        env_total = sum(self.t[k] for k in env_keys)
        print(
            f"\nenv.* (root-side game ops) total: {env_total:.2f}s "
            f"({env_total / total * 100:.1f}% of wall, "
            f"{(env_total / moves) * 1000:.1f} ms/move avg over {moves} lockstep moves, "
            f"N={n_games} games)"
        )


def install_net_timers(network, b: Buckets):
    orig_init = network.initial_inference
    orig_rec = network.recurrent_inference

    def init(*args, **kwargs):
        if b.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_init(*args, **kwargs)
        if b.sync_cuda:
            torch.cuda.synchronize()
        b.add("net.initial_inference", time.perf_counter() - t0)
        return out

    def rec(*args, **kwargs):
        if b.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_rec(*args, **kwargs)
        if b.sync_cuda:
            torch.cuda.synchronize()
        b.add("net.recurrent_inference", time.perf_counter() - t0)
        return out

    network.initial_inference = init
    network.recurrent_inference = rec


def play_batch_instrumented(network, game, config, n_games: int, device: str,
                            max_moves: int, b: Buckets) -> int:
    network.eval()
    mcts = BatchedMCTS(network, game, config, device)
    states = [game.reset() for _ in range(n_games)]
    histories = [GameHistory(game_name=config.game) for _ in range(n_games)]
    move_counts = [0] * n_games
    active = list(range(n_games))
    n_frames = getattr(config, "history_frames", 1)

    move_iter = 0
    while active and move_iter < max_moves:
        with b.timed("env.to_tensor"):
            single_frames = [game.to_tensor(states[g]) for g in active]
        with b.timed("env.legal_actions"):
            legal_list = [game.legal_actions(states[g]) for g in active]
        with b.timed("env.stack_history"):
            obs_list = [
                stack_with_history(single_frames[i], histories[g].observations, n_frames)
                for i, g in enumerate(active)
            ]

        with b.timed("mcts.run_batch"):
            roots = mcts.run_batch(obs_list, legal_list, add_noise=True)

        with b.timed("mcts.select_action"):
            sels = [select_action(r, temperature=1.0) for r in roots]

        with b.timed("env.step"):
            still_active = []
            for i, g in enumerate(active):
                action, action_probs = sels[i]
                histories[g].observations.append(single_frames[i])
                histories[g].actions.append(action)
                histories[g].policies.append(action_probs)
                histories[g].root_values.append(roots[i].value)
                histories[g].legal_actions_list.append(legal_list[i])
                state, reward, _ = game.step(states[g], action)
                histories[g].rewards.append(reward)
                states[g] = state
                move_counts[g] += 1
                if not state.done:
                    still_active.append(g)
            active = still_active

        move_iter += 1

    return move_iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-parallel-games", type=int, default=64,
                   help="N parallel games (production chess uses 256).")
    p.add_argument("--num-simulations", type=int, default=100,
                   help="MCTS simulations per move (production chess uses 400).")
    p.add_argument("--max-moves", type=int, default=20,
                   help="Lockstep moves to play in benchmark loop.")
    p.add_argument("--no-sync", action="store_true",
                   help="Disable per-bucket cuda.synchronize (faster but less accurate "
                        "attribution between net + mcts buckets).")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--use-gpu-chess", action="store_true",
                   help="Reserved for Phase 5; currently no-op.")
    args = p.parse_args()

    sync_cuda = (args.device == "cuda") and not args.no_sync

    config = get_config("chess")
    config.num_simulations = args.num_simulations
    config.num_parallel_games = args.num_parallel_games

    print(
        f"Bench: device={args.device} N={args.num_parallel_games} "
        f"sims={args.num_simulations} max_moves={args.max_moves} "
        f"history_frames={getattr(config, 'history_frames', 1)} "
        f"sample_k={getattr(config, 'sample_k', None)} "
        f"sync_cuda={sync_cuda}"
    )

    game = ChessGame()
    network = MuZeroNetwork(
        observation_channels=game.num_planes * getattr(config, "history_frames", 1),
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h,
        latent_w=config.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_consistency_loss=config.use_consistency_loss,
        proj_hid=config.proj_hid,
        proj_out=config.proj_out,
        pred_hid=config.pred_hid,
        pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
        value_head_type=getattr(config, "value_head_type", "support"),
        draw_score=getattr(config, "draw_score", 0.0),
    ).to(args.device)
    print(f"Network params: {sum(p.numel() for p in network.parameters()):,}")

    b = Buckets(sync_cuda=sync_cuda)
    install_net_timers(network, b)

    if not args.no_warmup:
        print("Warmup (3 moves)...")
        warm_b = Buckets(sync_cuda=sync_cuda)
        play_batch_instrumented(
            network, game, config, args.num_parallel_games, args.device,
            max_moves=3, b=warm_b,
        )

    # Reset buckets after warmup
    b.t.clear()
    b.n.clear()

    print("Benchmark...")
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    moves = play_batch_instrumented(
        network, game, config, args.num_parallel_games, args.device,
        max_moves=args.max_moves, b=b,
    )
    if sync_cuda:
        torch.cuda.synchronize()
    total = time.perf_counter() - t0

    print(
        f"\nWall: {total:.2f}s | lockstep moves: {moves} | "
        f"per-move wall: {total / moves * 1000:.1f}ms"
    )
    b.report(total, moves, args.num_parallel_games)


if __name__ == "__main__":
    main()
