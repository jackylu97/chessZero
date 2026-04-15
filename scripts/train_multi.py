"""Multi-game MuZero training entrypoint."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.training.multi_game_trainer import MultiGameTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Game MuZero")
    parser.add_argument("--games", type=str, nargs="+",
                        default=["tictactoe", "connect4"],
                        choices=["tictactoe", "connect4", "chess", "checkers"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--hidden-planes", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="runs/multi_game")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    trainer = MultiGameTrainer(
        game_names=args.games,
        hidden_planes=args.hidden_planes,
        num_blocks=args.num_blocks,
        training_steps=args.steps,
        device=device,
        log_dir=args.log_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
