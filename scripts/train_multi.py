"""Multi-game MuZero training entrypoint."""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.training.multi_game_trainer import MultiGameTrainer
from src.training.run_id import generate_run_id


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Game MuZero")
    parser.add_argument("--games", type=str, nargs="+",
                        default=["tictactoe", "connect4"],
                        choices=["tictactoe", "connect4", "chess", "checkers"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--hidden-planes", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: auto-generate YYYY_MM_DD_NNNN).")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    run_id = args.run_id or generate_run_id(
        Path(args.checkpoints_dir) / "multi_game",
        Path(args.log_dir) / "multi_game",
    )
    print(f"Run ID: {run_id}")
    print(f"  Checkpoints: {Path(args.checkpoints_dir) / 'multi_game' / run_id}")
    print(f"  TensorBoard: {Path(args.log_dir) / 'multi_game' / run_id}")

    trainer = MultiGameTrainer(
        game_names=args.games,
        run_id=run_id,
        hidden_planes=args.hidden_planes,
        num_blocks=args.num_blocks,
        training_steps=args.steps,
        device=device,
        log_dir=args.log_dir,
        checkpoints_dir=args.checkpoints_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
