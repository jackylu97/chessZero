"""Generate Stockfish self-play games to warmstart the MuZero chess replay buffer.

Each game is serialized as a ``GameHistory`` with:
  - policies:        one-hot on the move Stockfish chose
  - external_values: per-step Stockfish eval, converted to side-to-move POV in [-1, +1]
                     via value = 2 * sigmoid(cp / 400) - 1 (Lc0 convention); mate → ±1
  - root_values:     same as external_values (for diagnostics / compat)
  - rewards:         zeros except the terminal reward (matches ChessGame.step)
  - game_outcome:    +1 white win, -1 black win, 0 draw

Games are batched into pickled shards under ``--out-dir``. Load via
``ReplayBuffer.load_warmstart_games(glob('shard_*.pkl'))``.

Usage:
    sudo apt install stockfish
    .venv/bin/python scripts/generate_stockfish_games.py \\
        --out-dir data/stockfish_games/ \\
        --num-games 5000 --depth 8 --shard-size 500
"""

import argparse
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
import numpy as np

from src.games.chess import ChessGame, _move_to_action
from src.training.replay_buffer import GameHistory


CP_SCALE = 400.0            # Lc0 "soft saturation" — cp / 400 into sigmoid
MATE_VALUE = 1.0            # clip mate scores to ±1.0 (the support can accommodate it)


def score_to_value(score: chess.engine.PovScore, pov: chess.Color) -> float:
    """Convert Stockfish PovScore to a scalar in [-1, +1] from ``pov``'s perspective."""
    pov_score = score.pov(pov)
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        # mate() > 0 means pov delivers mate in N; < 0 means pov gets mated in N.
        return MATE_VALUE if mate_in > 0 else -MATE_VALUE
    cp = pov_score.score()
    if cp is None:
        return 0.0
    return 2.0 * (1.0 / (1.0 + math.exp(-cp / CP_SCALE))) - 1.0


def generate_one_game(
    engine: chess.engine.SimpleEngine,
    game: ChessGame,
    depth: int,
    max_plies: int,
) -> GameHistory | None:
    """Play a single Stockfish-vs-Stockfish game at fixed depth, recording per-move eval.

    Returns the GameHistory, or None if the game hit max_plies without a natural result
    (rare at depth 8 but possible — we drop these to keep label quality high).
    """
    board = chess.Board()
    state = game.reset()
    history = GameHistory(game_name="chess")

    while not state.done:
        if board.ply() >= max_plies:
            return None

        # Eval BEFORE the move (from the current side-to-move's POV).
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        value_stm = score_to_value(info["score"], board.turn)

        result = engine.play(board, chess.engine.Limit(depth=depth))
        move = result.move
        if move is None:
            return None

        # Record position + chosen action.
        obs = game.to_tensor(state)
        action = _move_to_action(move)
        one_hot = np.zeros(game.action_space_size, dtype=np.float32)
        one_hot[action] = 1.0

        history.observations.append(obs)
        history.actions.append(action)
        history.policies.append(one_hot)
        history.root_values.append(value_stm)
        history.external_values.append(value_stm)
        history.legal_actions_list.append(game.legal_actions(state))

        # Advance both the python-chess board (for engine.play) and ChessGame state
        # (for proper terminal reward). They must agree.
        board.push(move)
        state, reward, _ = game.step(state, action)
        history.rewards.append(reward)

    # Terminal observation + outcome.
    history.observations.append(game.to_tensor(state))
    history.game_outcome = float(state.winner)

    # Terminal external_value: side-to-move-relative ground truth.
    # After N actions, side to move = white if N even else black.
    # game_outcome is from p1 (white) POV, so flip for black.
    terminal_sign = 1.0 if (len(history.actions) % 2 == 0) else -1.0
    history.external_values.append(terminal_sign * history.game_outcome)

    return history


def save_shard(path: Path, games: list[GameHistory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default="stockfish", help="Stockfish binary (in PATH or absolute)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-games", type=int, default=5000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=400, help="drop games exceeding this")
    ap.add_argument("--stockfish-threads", type=int, default=1, help="UCI Threads option")
    ap.add_argument("--stockfish-hash", type=int, default=128, help="UCI Hash MB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        engine.configure({"Threads": args.stockfish_threads, "Hash": args.stockfish_hash})
    except chess.engine.EngineError as e:
        print(f"Warning: could not configure engine ({e}); continuing with defaults")

    game = ChessGame()

    shard: list[GameHistory] = []
    shard_idx = 0
    completed = 0
    dropped_cap = 0
    t0 = time.time()

    print(f"Generating {args.num_games} games at depth {args.depth} → {args.out_dir}")
    try:
        while completed < args.num_games:
            g = generate_one_game(engine, game, args.depth, args.max_plies)
            if g is None:
                dropped_cap += 1
                continue
            shard.append(g)
            completed += 1

            if len(shard) >= args.shard_size:
                path = args.out_dir / f"shard_{shard_idx:04d}.pkl"
                save_shard(path, shard)
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                decisive = sum(1 for gg in shard if gg.game_outcome != 0.0)
                mean_plies = np.mean([len(gg.actions) for gg in shard])
                print(f"  shard {shard_idx}: {len(shard)} games → {path} "
                      f"(total {completed}/{args.num_games}, "
                      f"{rate:.2f} g/s, decisive {decisive}/{len(shard)}, "
                      f"mean plies {mean_plies:.1f}, dropped {dropped_cap})")
                shard = []
                shard_idx += 1

        if shard:
            path = args.out_dir / f"shard_{shard_idx:04d}.pkl"
            save_shard(path, shard)
            print(f"  final shard {shard_idx}: {len(shard)} games → {path}")
    finally:
        engine.quit()

    total_time = time.time() - t0
    print(f"\nDone: {completed} games in {total_time/60:.1f} min "
          f"({completed/max(total_time, 1e-6):.2f} g/s), dropped {dropped_cap} over ply cap")


if __name__ == "__main__":
    main()
