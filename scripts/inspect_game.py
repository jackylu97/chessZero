"""Walk a played-out self-play game ply-by-ply.

Loads a game from a checkpoint's ``.buf`` file (or a single GameHistory
pickle) and dumps the position, picked move, MCTS visit distribution,
and the network's predictions for each ply.

Useful for "why did this game end in a draw?" forensics:

- Spot the moment the network "gives up" — i.e. switches from V≈+0.5
  to V≈0 on what's still a winning position.
- See if the visit distribution at the root is sharp (single dominant
  move) or diffuse (many candidates).
- For WDL heads, watch P(D) climb across plies — a draw-basin signature.

Examples:

  # Inspect game 0 from the latest .buf of a run.
  python scripts/inspect_game.py \\
      --buffer checkpoints/chess/2026_05_06_no_warmstart_fast/checkpoint_5000.buf \\
      --game-idx 0

  # Compare with current network's predictions on the same positions.
  python scripts/inspect_game.py --buffer <buf> --game-idx 0 \\
      --checkpoint <ckpt.pt>

  # Limit which plies to print (e.g. plies 30 onward).
  python scripts/inspect_game.py --buffer <buf> --game-idx 0 \\
      --start-ply 30 --max-plies 20
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.games.base import GameState
from src.training.replay_buffer import GameHistory, ReplayBuffer, stack_with_history

from scripts.inspect_mcts import load_checkpoint


def load_game(path: str, idx: int, game) -> GameHistory:
    """Load a single GameHistory from a .buf streaming file or a .pkl/.pickle file."""
    p = Path(path)
    if p.suffix in (".pkl", ".pickle"):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, GameHistory):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], GameHistory):
            return obj[idx]
        raise ValueError(f"Unsupported pickle contents at {path}")
    # .buf streaming.
    rb = ReplayBuffer(max_size=10_000_000)
    rb.load(p, game=game)
    if idx < 0 or idx >= len(rb.buffer):
        raise IndexError(f"game-idx {idx} out of range [0, {len(rb.buffer)})")
    return rb.buffer[idx]


def reconstruct_board_at(game_hist: GameHistory, ply: int) -> chess.Board:
    """Replay actions 0..ply-1 from the start position to get the board at ply."""
    board = chess.Board()
    for i in range(ply):
        action = int(game_hist.actions[i])
        move = _action_to_move(action, board)
        if move is None or move not in board.legal_moves:
            raise ValueError(
                f"reconstruction failed at ply {i}: action {action} is illegal "
                f"({move.uci() if move else 'None'}) on {board.fen()}"
            )
        board.push(move)
    return board


def _topk_action_summary(policy: np.ndarray, board: chess.Board, k: int = 5) -> str:
    """Render top-K actions from a stored policy ([A] dense numpy array)."""
    if policy is None or len(policy) == 0:
        return "<no policy>"
    nonzero = np.argsort(-policy)[:k]
    items: list[str] = []
    for a in nonzero:
        prob = float(policy[a])
        if prob <= 0:
            break
        m = _action_to_move(int(a), board)
        if m is None:
            items.append(f"<bad {a}>:{prob:.2f}")
            continue
        try:
            san = board.san(m) if m in board.legal_moves else f"<illegal {m.uci()}>"
        except Exception:
            san = m.uci()
        items.append(f"{san}({prob:.2f})")
    return ", ".join(items) if items else "<all-zero>"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--buffer", required=True,
                   help="Path to .buf (streaming) or .pkl GameHistory file.")
    p.add_argument("--game-idx", type=int, default=0,
                   help="Index of the game in the buffer (default 0).")
    p.add_argument("--start-ply", type=int, default=0)
    p.add_argument("--max-plies", type=int, default=0,
                   help="Limit number of plies printed (0 = all from start-ply onward).")
    p.add_argument("--checkpoint", default=None,
                   help="Optional checkpoint .pt; if given, also runs the "
                        "current network's initial_inference on each position "
                        "and prints its V_pred + WDL alongside the stored values.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K stored-policy actions to print per ply.")
    args = p.parse_args()

    config = get_config("chess")
    game = ChessGame()

    # Load game.
    game_hist = load_game(args.buffer, args.game_idx, game)
    n = len(game_hist.actions)
    print(f"Loaded game {args.game_idx} from {args.buffer}")
    print(f"  plies: {n}")
    print(f"  outcome: {game_hist.game_outcome:+.0f}  (1=W win, -1=B win, 0=draw)")
    print(f"  external_values (warmstart): {bool(getattr(game_hist, 'external_values', None))}")
    print()

    # Optional: load checkpoint to compare.
    network = None
    if args.checkpoint:
        network, step = load_checkpoint(args.checkpoint, config, game, device=args.device)
        print(f"Loaded checkpoint at step {step} for live network probe ({args.checkpoint})")
        print()

    n_frames = int(getattr(config, "history_frames", 1))
    vh = getattr(config, "value_head_type", "support")

    end_ply = n if args.max_plies == 0 else min(n, args.start_ply + args.max_plies)
    board = reconstruct_board_at(game_hist, args.start_ply)

    for ply in range(args.start_ply, end_ply):
        action = int(game_hist.actions[ply])
        move = _action_to_move(action, board)
        if move is None or move not in board.legal_moves:
            print(f"--- ply {ply} ---  invalid action {action}: {move}")
            break
        san = board.san(move)
        stored_v = float(game_hist.root_values[ply]) if ply < len(game_hist.root_values) else float("nan")
        reward = float(game_hist.rewards[ply]) if ply < len(game_hist.rewards) else float("nan")
        policy = game_hist.policies[ply] if ply < len(game_hist.policies) else None
        if policy is not None:
            policy = np.asarray(policy, dtype=np.float64)
            if policy.size:
                top_actions = _topk_action_summary(policy, board, k=args.top_k)
                top_action_idx = int(np.argmax(policy))
                pol_entropy = float(-(policy * np.log(np.clip(policy, 1e-12, None))).sum())
            else:
                top_actions = "<empty>"
                pol_entropy = float("nan")
        else:
            top_actions = "<no policy>"
            pol_entropy = float("nan")

        # Header line: ply, mover, played move, picked-vs-top, stored value.
        mover = "W" if board.turn == chess.WHITE else "B"
        pick_marker = "≡" if (policy is not None and policy.size and top_action_idx == action) else "≠"
        print(
            f"ply {ply:3d} ({mover}) — played {san:6s} {pick_marker}argmax(π)  "
            f"V_stored={stored_v:+.3f}  H(π)={pol_entropy:5.2f}  reward={reward:+.2f}"
        )
        print(f"  fen: {board.fen()}")
        print(f"  top-{args.top_k}: {top_actions}")

        if network is not None:
            # Reconstruct the T-frame stacked obs as MCTS would have seen.
            obs_history = game_hist.observations[: ply + 1]
            single = obs_history[ply] if ply < len(obs_history) else game.to_tensor(
                GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1, done=False, winner=0)
            )
            stacked = stack_with_history(single, obs_history[:ply], n_frames)
            obs_batch = stacked.unsqueeze(0).to(args.device)
            with torch.no_grad():
                hidden, policy_logits, value_root = network.initial_inference(obs_batch)
                v_now = float(value_root.squeeze())
                if vh == "wdl":
                    _, value_logits = network.prediction(hidden)
                    wdl = F.softmax(value_logits, dim=-1).squeeze(0)
                    pw, pd, pl = float(wdl[0]), float(wdl[1]), float(wdl[2])
                    print(f"  net_now: V_pred={v_now:+.3f}  W/D/L={pw:.2f}/{pd:.2f}/{pl:.2f}")
                else:
                    print(f"  net_now: V_pred={v_now:+.3f}")

        board.push(move)
        print()


if __name__ == "__main__":
    main()
