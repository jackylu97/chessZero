"""Main training entrypoint for MuZero."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import get_config
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork
from src.training.trainer import MuZeroTrainer


GAMES = {
    "tictactoe": TicTacToe,
}

# Lazy imports for games that may not be needed
def get_game(name: str):
    if name in GAMES:
        return GAMES[name]()
    if name == "connect4":
        from src.games.connect4 import Connect4
        return Connect4()
    if name == "chess":
        from src.games.chess import ChessGame
        return ChessGame()
    if name == "checkers":
        from src.games.checkers import Checkers
        return Checkers()
    raise ValueError(f"Unknown game: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train MuZero")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "connect4", "chess", "checkers"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    config = get_config(args.game)
    config.device = device
    if args.steps is not None:
        config.training_steps = args.steps

    # Use CPU AMP settings appropriately
    if device == "cpu":
        config.use_amp = False

    game = get_game(args.game)

    network = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h,
        latent_w=config.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
    )

    trainer = MuZeroTrainer(config, game, network, device, args.log_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.network.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.global_step = checkpoint["step"]
        print(f"Resumed from step {checkpoint['step']}")

    trainer.train()


if __name__ == "__main__":
    main()
