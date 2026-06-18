"""Generate Stockfish-solved elementary-endgame games (KQ/KR/KP vs K) as a
conversion curriculum, in the warmstart-injection shard format (format_version 2).

Why: the model can't convert won endgames (shuffles KQ-v-K for 300+ plies), and the
regular warmstart barely covers this — only ~5% of warmstart games ever reach a bare
elementary endgame (strong Stockfish games end with material still on the board). These
seeds flood the buffer with crisp conversions: SHARP policy targets pointing at the
mating move, short games, so the policy learns technique and the moves-left head learns
"these end fast." Mix into the injection pool so they persist into self-play (the anchor).

Run: .venv/bin/python scripts/generate_endgame_games.py --out data/endgame_seed \
       --per-type 500 --label-depth 14 --tau-label 0.1
"""
import argparse, os, sys, random
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess, chess.engine
import numpy as np
from src.games.base import GameState
from src.games.chess import ChessGame, _move_to_action
from src.training.replay_buffer import GameHistory
from scripts.generate_stockfish_games import (
    score_to_value, softmax_sample_from_multipv, save_shard,
)

# Strong side = white. Skip KBN-v-K: Stockfish at practical depth often can't mate it
# within the 50-move rule (verified — drew 3/4 at depth 12).
SPECS = {
    "KQvK": [chess.QUEEN],
    "KRvK": [chess.ROOK],
    "KPvK": [chess.PAWN],
}


def rand_board(piece_types, rng):
    """Random legal position: 2 kings + white non-king material, white to move."""
    for _ in range(3000):
        sqs = rng.sample(range(64), 2 + len(piece_types))
        b = chess.Board.empty()
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        bad = False
        for pt, sq in zip(piece_types, sqs[2:]):
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                bad = True; break
            b.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
        if bad:
            continue
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_game_over():
            return b
    return None


def play_out(start_board, game, engine, label_depth, multipv, tau_label, max_plies):
    """Stockfish plays the strong side's conversion (deterministic best move),
    labels every ply (value + SHARP soft policy). Returns GameHistory or None."""
    board = start_board.copy()
    state = GameState(board=board.copy(),
                      current_player=1 if board.turn == chess.WHITE else -1)
    h = GameHistory(game_name="chess")
    h.start_fen = start_board.fen()  # games begin from this endgame, NOT the standard start
    while not state.done:
        if board.ply() >= max_plies:
            return None
        info = engine.analyse(board, chess.engine.Limit(depth=label_depth), multipv=multipv)
        if isinstance(info, dict):
            info = [info]
        if not info or not info[0].get("pv"):
            return None
        label_value = score_to_value(info[0]["score"], board.turn)
        move, _, _ = softmax_sample_from_multipv(info, board.turn, 0.0, None)  # deterministic best
        if move is None:
            return None
        action = _move_to_action(move, board.turn)
        # CRISP conversion target: one-hot on Stockfish's best (fastest-mate) move.
        # A multipv soft target collapses to uniform here because every mating move
        # clips to +1.0 in score_to_value (mate-in-3 indistinguishable from mate-in-7)
        # -> diffuse, which is the exact pathology we're trying to teach the model OUT of.
        policy = np.zeros(game.action_space_size, dtype=np.float32)
        policy[action] = 1.0
        h.observations.append(game.to_tensor(state))
        h.actions.append(action)
        h.policies.append(policy)
        h.root_values.append(label_value)
        h.external_values.append(label_value)
        h.legal_actions_list.append(game.legal_actions(state))
        board.push(move)
        state, reward, _ = game.step(state, action)
        h.rewards.append(reward)
    h.observations.append(game.to_tensor(state))
    h.game_outcome = float(state.winner)
    terminal_sign = 1.0 if (len(h.actions) % 2 == 0) else -1.0
    h.external_values.append(terminal_sign * h.game_outcome)
    h.draw_by_repetition = bool(board.is_repetition(3))
    h.draw_by_no_progress = bool(board.halfmove_clock >= 100)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/endgame_seed")
    ap.add_argument("--per-type", type=int, default=500, help="WON games kept per endgame type")
    ap.add_argument("--label-depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--tau-label", type=float, default=0.1, help="low = sharp (crisp technique)")
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    game = ChessGame()
    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    kept, attempts, total_plies, shard_idx = [], 0, 0, 0
    per_type_kept = {k: 0 for k in SPECS}
    try:
        for name, pts in SPECS.items():
            while per_type_kept[name] < args.per_type:
                attempts += 1
                b = rand_board(pts, rng)
                if b is None:
                    continue
                h = play_out(b, game, eng, args.label_depth, args.multipv, args.tau_label, args.max_plies)
                if h is None or h.game_outcome <= 0:   # keep only WON-for-strong-side conversions
                    continue
                kept.append(h); per_type_kept[name] += 1; total_plies += len(h.actions)
                if len(kept) >= args.shard_size:
                    p = Path(args.out) / f"endgame_shard_{shard_idx:04d}.pkl"
                    save_shard(p, kept, format_version=2); shard_idx += 1
                    print(f"  wrote {p} ({len(kept)} games)", flush=True)
                    kept = []
            print(f"{name}: kept {per_type_kept[name]} (attempts so far {attempts})", flush=True)
        if kept:
            p = Path(args.out) / f"endgame_shard_{shard_idx:04d}.pkl"
            save_shard(p, kept, format_version=2)
            print(f"  wrote {p} ({len(kept)} games)", flush=True)
    finally:
        eng.quit()
    n = sum(per_type_kept.values())
    print(f"\nDONE: {n} won-conversion games -> {args.out}  "
          f"(mean {total_plies/max(1,n):.0f} plies, {attempts} attempts, keep-rate {n/max(1,attempts):.0%})")


if __name__ == "__main__":
    main()
