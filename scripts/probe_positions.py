"""Probe network's raw value/policy on a curated FEN bench.

Loads a checkpoint and runs ``network.initial_inference`` on a list of
test positions categorized by expected outcome. Prints a table showing
the network's predicted value, (for WDL heads) W/D/L distribution,
policy entropy, and the network's top-3 actions.

Useful for diagnosing the value head's calibration over training:

- **Tactical-winning** positions should drift toward V=+1 over checkpoints.
- **Drawn** endgames should show V≈0 with high P(D).
- **Quiet** positions should produce non-degenerate values that vary
  across positions (not all collapsed to 0).
- **Losing** positions should drift to V=-1 for STM.

If V≈0 and P(D)≈1 across all categories, the value head has collapsed
to predict-draw — the canonical draw-basin signature in raw form.

Examples:

  # Default FEN bench, latest checkpoint of a run.
  python scripts/probe_positions.py \\
      --checkpoint checkpoints/chess/2026_05_06_no_warmstart_fast/checkpoint_5000.pt

  # Custom FEN list (pipe-separated: category|label|fen).
  python scripts/probe_positions.py --checkpoint <ckpt.pt> \\
      --fens-file my_positions.txt

  # Sweep checkpoints from a run (bash):
  for ckpt in checkpoints/chess/<run-id>/checkpoint_*.pt; do
      step=$(basename $ckpt .pt | sed 's/checkpoint_//')
      echo "=== step $step ==="
      python scripts/probe_positions.py --checkpoint "$ckpt"
  done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import torch
import torch.nn.functional as F

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.games.base import GameState
from src.training.replay_buffer import stack_with_history

from scripts.inspect_mcts import load_checkpoint


# Default FEN bench. Each entry: (category, label, fen).
# Categories help interpretation:
#   opening       — should be near 0 (untrained: exactly 0; trained: small bias).
#   tactical_win  — STM has a forced winning sequence; trained net should give V→+1.
#   quiet         — no immediate tactic; trained net should give differentiated values.
#   draw          — objectively drawn endgames; should give V≈0 with high P(D) (WDL).
#   losing        — STM is materially down; trained net should give V→-1.
DEFAULT_FENS = [
    # Openings.
    ("opening", "startpos",                          chess.STARTING_FEN),
    ("opening", "after 1.e4",                        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
    ("opening", "after 1.e4 e5 2.Nf3",               "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
    ("opening", "Sicilian Open after Nxd4",          "rnbqkbnr/pp1ppppp/8/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 3"),

    # Tactical wins (chosen so STM has a clear forced win in 1-3 moves).
    ("tactical_win", "Back-rank M1 white",           "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
    ("tactical_win", "Smothered M2 white",           "6rk/6pp/8/8/8/8/8/4K1NR w K - 0 1"),
    ("tactical_win", "Knight fork M3 white",         "r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1"),
    ("tactical_win", "Black M1 (queen pin)",         "r3k2r/ppp2ppp/8/8/8/8/PPPP1PPP/4Kq2 w - - 0 1"),

    # Quiet middlegames — strong engines have non-trivial evals here.
    ("quiet", "Italian middle",                      "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"),
    ("quiet", "QGD Orthodox",                        "r1bq1rk1/ppp1bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7"),
    ("quiet", "King's Indian Saemisch",              "rnbqkb1r/pppppp1p/5np1/8/3PP3/2N1P3/PPP2P1P/R1BQKBNR b KQkq - 0 4"),
    ("quiet", "Caro-Kann Classical",                 "r1bqkbnr/pp1npppp/2p5/8/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 1 5"),

    # Drawn endgames.
    ("draw", "K vs K",                               "8/8/8/8/4k3/8/8/4K3 w - - 0 1"),
    ("draw", "K+B vs K (insufficient)",              "8/8/8/8/4k3/8/8/4KB2 w - - 0 1"),
    ("draw", "K+N vs K (insufficient)",              "8/8/8/8/4k3/8/8/4KN2 w - - 0 1"),
    ("draw", "K+P vs K (drawn opposition)",          "8/8/8/8/3k4/8/3KP3/8 b - - 0 1"),

    # Clearly losing for STM (down material, no compensation).
    ("losing", "Down a queen, white to move",        "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBN1 w KQkq - 0 1"),
    ("losing", "Down a rook, black to move",         "rnbqkbnr/ppp2ppp/3pp3/8/3PP3/8/PPP2PPP/R1BQKBNR b KQkq - 0 1"),
]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--fens-file", default=None,
                   help="Optional file with custom FENs (pipe-separated: "
                        "'category|label|fen'). Lines starting with # ignored.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    config = get_config("chess")
    game = ChessGame()
    network, step = load_checkpoint(args.checkpoint, config, game, device=args.device)
    print(f"Loaded checkpoint at step {step} ({args.checkpoint})")
    print(f"  value_head={getattr(config, 'value_head_type', 'support')}, "
          f"draw_score={getattr(config, 'draw_score', 0.0)}")
    print()

    fens = DEFAULT_FENS
    if args.fens_file:
        custom = []
        with open(args.fens_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|", 2)
                if len(parts) != 3:
                    continue
                custom.append(tuple(s.strip() for s in parts))
        fens = custom

    n_frames = int(getattr(config, "history_frames", 1))
    vh = getattr(config, "value_head_type", "support")

    # Header.
    if vh == "wdl":
        print(f"{'category':<14}{'label':<26}{'V_pred':>8}  {'P_W':>5} {'P_D':>5} {'P_L':>5}  {'H(π)':>6}  top-3")
    else:
        print(f"{'category':<14}{'label':<26}{'V_pred':>8}  {'H(π)':>6}  top-3")
    print("-" * 100)

    # Aggregate stats per category.
    by_cat: dict[str, list[float]] = {}

    for category, label, fen in fens:
        try:
            board = chess.Board(fen)
        except Exception as e:
            print(f"{category:<14}{label[:24]:<26}  invalid FEN: {e}")
            continue
        state = GameState(
            board=board,
            current_player=1 if board.turn == chess.WHITE else -1,
            done=False,
            winner=0,
        )
        single = game.to_tensor(state)
        stacked = stack_with_history(single, [], n_frames)
        obs = stacked.unsqueeze(0).to(args.device)

        with torch.no_grad():
            hidden, policy_logits, value_root = network.initial_inference(obs)
            policy = F.softmax(policy_logits, dim=-1).squeeze(0)
            entropy = float(-(policy * policy.clamp(min=1e-12).log()).sum())
            v = float(value_root.squeeze())
            if vh == "wdl":
                _, value_logits = network.prediction(hidden)
                wdl = F.softmax(value_logits, dim=-1).squeeze(0)
                pw, pd, pl = float(wdl[0]), float(wdl[1]), float(wdl[2])

        # Top-3 by network policy among legal moves.
        legal = game.legal_actions(state)
        if legal:
            mask = torch.zeros_like(policy)
            mask[legal] = 1.0
            top3 = torch.topk(policy * mask, k=min(3, len(legal))).indices.cpu().tolist()
            top3_san: list[str] = []
            for a in top3:
                m = _action_to_move(a, board)
                if m is not None and m in board.legal_moves:
                    try:
                        top3_san.append(board.san(m))
                    except Exception:
                        top3_san.append(m.uci())
            top_str = ", ".join(top3_san) if top3_san else "<none>"
        else:
            top_str = "<no legal>"

        if vh == "wdl":
            print(
                f"{category:<14}{_truncate(label, 26):<26}{v:>+8.3f}  "
                f"{pw:5.2f} {pd:5.2f} {pl:5.2f}  {entropy:6.2f}  {top_str}"
            )
        else:
            print(f"{category:<14}{_truncate(label, 26):<26}{v:>+8.3f}  {entropy:6.2f}  {top_str}")
        by_cat.setdefault(category, []).append(v)

    print()
    print("=== per-category mean V_pred ===")
    for cat, values in by_cat.items():
        n = len(values)
        mean = sum(values) / n
        std = (sum((x - mean) ** 2 for x in values) / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0
        print(f"  {cat:<14}n={n:>2}  mean={mean:+.3f}  std={std:.3f}")


if __name__ == "__main__":
    main()
