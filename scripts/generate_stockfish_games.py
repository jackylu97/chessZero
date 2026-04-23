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
from typing import Tuple

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


def softmax_sample_from_multipv(
    info_list: list,
    pov: chess.Color,
    temperature: float,
    rng: random.Random,
) -> Tuple[chess.Move | None, float, float]:
    """Sample a move from MultiPV analysis, weighted by softmax over line evals.

    Returns ``(move, chosen_eval, best_eval)`` where:
      - ``chosen_eval`` is ``info_list[k]['score']`` in ``pov`` POV for the sampled k —
        the correct value target (eval of the line we're actually playing).
      - ``best_eval`` is the top-line eval, returned so the caller can log
        quality (how far below best we sampled).

    At ``temperature <= 0`` or len(moves) == 1, deterministically returns top-1.
    Numerical stability: subtracts eval max before exp.
    """
    moves = [info["pv"][0] for info in info_list if info.get("pv")]
    if not moves:
        return None, 0.0, 0.0
    evals = np.array(
        [score_to_value(info["score"], pov) for info in info_list[: len(moves)]],
        dtype=np.float64,
    )
    best_eval = float(evals[0])

    if temperature <= 0.0 or len(moves) == 1:
        return moves[0], float(evals[0]), best_eval

    w = np.exp((evals - evals.max()) / temperature)
    w /= w.sum()
    idx = rng.choices(range(len(moves)), weights=w.tolist())[0]
    return moves[idx], float(evals[idx]), best_eval


def generate_one_game(
    engine: chess.engine.SimpleEngine,
    game: ChessGame,
    depth: int,
    max_plies: int,
    multipv: int,
    move_temperature: float,
    rng: random.Random,
    gap_sink: list | None = None,
) -> GameHistory | None:
    """Play a single Stockfish game at fixed depth with softmax-over-top-K move sampling.

    Returns the GameHistory, or None if the game hit max_plies without a natural result.
    If ``gap_sink`` is provided, per-ply ``(best_eval - chosen_eval)`` gaps are appended
    for diversity diagnostics.
    """
    board = chess.Board()
    state = game.reset()
    history = GameHistory(game_name="chess")

    while not state.done:
        if board.ply() >= max_plies:
            return None

        info_list = engine.analyse(
            board, chess.engine.Limit(depth=depth), multipv=multipv
        )
        if isinstance(info_list, dict):
            info_list = [info_list]

        move, value_stm, best_eval = softmax_sample_from_multipv(
            info_list, board.turn, move_temperature, rng
        )
        if move is None:
            return None
        if gap_sink is not None:
            gap_sink.append(best_eval - value_stm)

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

        # Advance both the python-chess board (for engine state) and ChessGame
        # state (for proper terminal reward). They must agree.
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


def generate_one_asymmetric_game(
    engine: chess.engine.SimpleEngine,
    game: ChessGame,
    play_depth_white: int,
    play_depth_black: int,
    label_depth: int,
    max_plies: int,
    multipv: int,
    move_temperature: float,
    rng: random.Random,
    gap_sink: list | None = None,
) -> GameHistory | None:
    """Asymmetric-teacher game: play at mixed depths, label everything at label_depth.

    At each ply, two analyse calls (collapsed to one when play_depth == label_depth):
      1. label_info @ depth=label_depth, MPV=1 — drives `policies` (one-hot on top move)
         and `external_values` / `root_values` (STM eval).
      2. play_info  @ depth=play_depth_{w,b}, MPV=multipv — drives `actions` (the move
         actually pushed to the board) via softmax-over-top-K sampling.

    ``policies[i]`` and ``actions[i]`` diverge on positions where the weaker play
    engine would pick a different move than depth-8's preference. That divergence
    IS the tactical-signal this is designed to create — depth-8 labels every
    position accurately, and the weak-side trajectory puts bad positions in the
    training set so dynamics learns the cause-and-effect.
    """
    board = chess.Board()
    state = game.reset()
    history = GameHistory(game_name="chess")

    while not state.done:
        if board.ply() >= max_plies:
            return None

        play_depth = play_depth_white if board.turn == chess.WHITE else play_depth_black

        if play_depth == label_depth:
            # Single analyse: play engine at label depth with MPV=K covers both uses.
            play_info = engine.analyse(
                board, chess.engine.Limit(depth=label_depth), multipv=multipv
            )
            if isinstance(play_info, dict):
                play_info = [play_info]
            label_info = [play_info[0]]
        else:
            label_info = engine.analyse(
                board, chess.engine.Limit(depth=label_depth), multipv=1
            )
            if isinstance(label_info, dict):
                label_info = [label_info]
            play_info = engine.analyse(
                board, chess.engine.Limit(depth=play_depth), multipv=multipv
            )
            if isinstance(play_info, dict):
                play_info = [play_info]

        # Label from depth-8 top line.
        if not label_info or not label_info[0].get("pv"):
            return None
        label_move = label_info[0]["pv"][0]
        label_value_stm = score_to_value(label_info[0]["score"], board.turn)

        # Play via softmax sample over depth-play_depth top-K.
        played_move, chosen_eval, best_eval = softmax_sample_from_multipv(
            play_info, board.turn, move_temperature, rng
        )
        if played_move is None:
            return None
        if gap_sink is not None:
            gap_sink.append(best_eval - chosen_eval)

        label_action = _move_to_action(label_move)
        played_action = _move_to_action(played_move)
        one_hot = np.zeros(game.action_space_size, dtype=np.float32)
        one_hot[label_action] = 1.0

        history.observations.append(game.to_tensor(state))
        history.actions.append(played_action)          # behavior (played) — may diverge
        history.policies.append(one_hot)                # target (labeled at depth-8)
        history.root_values.append(label_value_stm)
        history.external_values.append(label_value_stm)
        history.legal_actions_list.append(game.legal_actions(state))

        board.push(played_move)
        state, reward, _ = game.step(state, played_action)
        history.rewards.append(reward)

    history.observations.append(game.to_tensor(state))
    history.game_outcome = float(state.winner)
    terminal_sign = 1.0 if (len(history.actions) % 2 == 0) else -1.0
    history.external_values.append(terminal_sign * history.game_outcome)
    return history


# Bucket-name → (strong_depth, weak_depth). Strong always 8; weak varies.
BUCKETS = {
    "8v5": (8, 5),
    "8v6": (8, 6),
    "8v7": (8, 7),
    "8v8": (8, 8),
}


def save_shard(path: Path, games: list[GameHistory], format_version: int = 2) -> None:
    """Write a shard to disk.

    format_version:
        1 — legacy: a single pickled ``list[GameHistory]`` (~2.8 MB/game).
        2 — compact streaming: header dict then one compact-dict per game
            (~10-50 KB/game for Stockfish one-hots). Default.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_version == 1:
        with open(path, "wb") as f:
            pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    if format_version != 2:
        raise ValueError(f"Unsupported shard format_version: {format_version}")
    header = {"version": 2, "n_records": len(games)}
    with open(path, "wb") as f:
        pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for g in games:
            pickle.dump(g.to_compact_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)


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
    ap.add_argument("--shard-index-start", type=int, default=0,
                    help="Starting shard index (for running parallel workers writing "
                         "to the same --out-dir with disjoint index ranges).")
    ap.add_argument("--multipv", type=int, default=3,
                    help="Top-K moves Stockfish returns per position. 1 = deterministic "
                         "best move (no diversity).")
    ap.add_argument("--move-temperature", type=float, default=0.15,
                    help="Softmax temperature τ over the top-K line evals in [-1,+1] "
                         "space. τ→0 = always best move; τ→∞ = uniform across top-K. "
                         "0.15 ≈ random in flat positions, deterministic on mate threats.")
    ap.add_argument("--format-version", type=int, default=2, choices=[1, 2],
                    help="Shard format. 1 = legacy list[GameHistory] (~2.8 MB/game); "
                         "2 = compact streaming (~10-50 KB/game, default). Training "
                         "loads both transparently.")
    ap.add_argument("--bucket", choices=list(BUCKETS.keys()), default=None,
                    help="Asymmetric-teacher bucket. 8v5/8v6/8v7 = depth-8 vs weak engine; "
                         "8v8 = depth-8 both sides (labels + K-way play sampling). Within "
                         "a bucket, colors alternate 50/50 per game so depth-8 plays white "
                         "half the time. Overrides --depth; labels at --label-depth (default 8).")
    ap.add_argument("--play-depth-white", type=int, default=None,
                    help="Explicit per-color play depth (advanced). Requires --play-depth-black. "
                         "Incompatible with --bucket.")
    ap.add_argument("--play-depth-black", type=int, default=None)
    ap.add_argument("--label-depth", type=int, default=8,
                    help="Evaluator depth used for policy/value labels in asymmetric mode "
                         "(default 8, matches --depth for backward compat of symmetric runs).")
    args = ap.parse_args()

    # Mode selection: bucket > explicit play-depth pair > symmetric --depth fallback.
    asymmetric = args.bucket is not None or args.play_depth_white is not None
    if args.bucket and args.play_depth_white is not None:
        ap.error("--bucket and --play-depth-white are mutually exclusive")
    if args.play_depth_white is not None and args.play_depth_black is None:
        ap.error("--play-depth-white requires --play-depth-black")

    if args.bucket:
        strong_depth, weak_depth = BUCKETS[args.bucket]

        def depths_for_game(i: int) -> Tuple[int, int]:
            # Even games: depth-8 white; odd games: depth-8 black. Alternation gives a
            # 50/50 color split per bucket so the model sees both sides of the asymmetry.
            if i % 2 == 0:
                return strong_depth, weak_depth
            return weak_depth, strong_depth
    elif asymmetric:
        w, b = args.play_depth_white, args.play_depth_black

        def depths_for_game(i: int) -> Tuple[int, int]:
            return w, b
    else:
        def depths_for_game(i: int) -> Tuple[int, int]:
            return args.depth, args.depth

    random.seed(args.seed)
    np.random.seed(args.seed)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        engine.configure({"Threads": args.stockfish_threads, "Hash": args.stockfish_hash})
    except chess.engine.EngineError as e:
        print(f"Warning: could not configure engine ({e}); continuing with defaults")

    game = ChessGame()
    rng = random.Random(args.seed)

    shard: list[GameHistory] = []
    shard_idx = args.shard_index_start
    completed = 0
    dropped_cap = 0
    t0 = time.time()

    if asymmetric:
        mode_str = (f"asymmetric bucket={args.bucket}"
                    if args.bucket else
                    f"asymmetric play_w={args.play_depth_white},b={args.play_depth_black}")
        print(f"Generating {args.num_games} games [{mode_str}] "
              f"label_depth={args.label_depth} multipv={args.multipv} "
              f"τ={args.move_temperature} → {args.out_dir}")
    else:
        print(f"Generating {args.num_games} games at depth {args.depth} "
              f"(multipv={args.multipv}, τ={args.move_temperature}) → {args.out_dir}")
    try:
        while completed < args.num_games:
            if asymmetric:
                pd_w, pd_b = depths_for_game(completed)
                g = generate_one_asymmetric_game(
                    engine, game,
                    play_depth_white=pd_w,
                    play_depth_black=pd_b,
                    label_depth=args.label_depth,
                    max_plies=args.max_plies,
                    multipv=args.multipv,
                    move_temperature=args.move_temperature,
                    rng=rng,
                )
            else:
                g = generate_one_game(
                    engine, game, args.depth, args.max_plies,
                    args.multipv, args.move_temperature, rng,
                )
            if g is None:
                dropped_cap += 1
                continue
            shard.append(g)
            completed += 1

            if len(shard) >= args.shard_size:
                path = args.out_dir / f"shard_{shard_idx:04d}.pkl"
                save_shard(path, shard, format_version=args.format_version)
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
            save_shard(path, shard, format_version=args.format_version)
            print(f"  final shard {shard_idx}: {len(shard)} games → {path}")
    finally:
        engine.quit()

    total_time = time.time() - t0
    print(f"\nDone: {completed} games in {total_time/60:.1f} min "
          f"({completed/max(total_time, 1e-6):.2f} g/s), dropped {dropped_cap} over ply cap")


if __name__ == "__main__":
    main()
