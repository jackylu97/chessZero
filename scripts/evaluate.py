"""Evaluate a trained MuZero model or play against it."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import get_config
from src.mcts.mcts import MCTS, select_action


def get_game(name: str):
    if name == "tictactoe":
        from src.games.tictactoe import TicTacToe
        return TicTacToe()
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


def load_model(checkpoint_path: str, game, config, device: str):
    from src.model.muzero_net import MuZeroNetwork
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
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
    )
    from src.config import MuZeroConfig
    torch.serialization.add_safe_globals([MuZeroConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device)
    network.eval()
    return network


def parse_tictactoe_move(move_str: str) -> int | None:
    """Parse 'row col' or 'row,col' (1-indexed) into a flat action index."""
    move_str = move_str.strip().replace(",", " ")
    parts = move_str.split()
    if len(parts) != 2:
        return None
    try:
        row, col = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (1 <= row <= 3 and 1 <= col <= 3):
        return None
    return (row - 1) * 3 + (col - 1)


def play_interactive(game, network, config, device):
    """Play interactively against the trained model."""
    from src.games.tictactoe import TicTacToe
    from src.games.chess import ChessGame
    is_tictactoe = isinstance(game, TicTacToe)
    is_chess = isinstance(game, ChessGame)

    mcts = MCTS(network, game, config, device)
    state = game.reset()

    if is_chess:
        print("You are Black. Model is White.")
        print("Enter moves as UCI (e2e4, g1f3, e7e8q) or SAN (e4, Nf3, O-O).")
        print("Type 'quit' to exit.\n")
    elif is_tictactoe:
        print("You are O (player -1). Model is X (player 1).")
        print("Enter moves as 'row col' (e.g. '2 3' for row 2, column 3).\n")
    else:
        print("You are player -1. Model is player 1.\n")

    if is_tictactoe:
        print(game.render_with_moves(state))
    else:
        print(game.render(state))

    while not state.done:
        legal = game.legal_actions(state)
        if state.current_player == 1:
            obs = game.to_tensor(state)
            root = mcts.run(obs, legal, add_noise=False)
            action, _ = select_action(root, temperature=0)
            if is_chess:
                print(f"\nModel plays: {game.action_to_san(state, action)}")
            elif is_tictactoe:
                row, col = divmod(action, 3)
                print(f"\nModel plays: row {row + 1}, col {col + 1}")
            else:
                print(f"\nModel plays: {action}")
        else:
            while True:
                try:
                    if is_chess:
                        raw = input("\nYour move: ").strip()
                        if raw.lower() in ("quit", "exit", "q"):
                            print("Bye.")
                            return
                        action = game.parse_human_move(state, raw)
                        if action is None:
                            print("Couldn't parse or not legal. Try UCI (e2e4) or SAN (Nf3).")
                            continue
                    elif is_tictactoe:
                        raw = input("\nYour move (row col): ")
                        action = parse_tictactoe_move(raw)
                        if action is None:
                            print("Enter row and column as two numbers, e.g. '2 3'.")
                            continue
                    else:
                        action = int(input(f"\nLegal moves: {legal}\nYour move: "))
                    if action in legal:
                        break
                    if is_chess:
                        print("That move isn't legal from this position.")
                    else:
                        print("That square is taken — pick an empty one.")
                except ValueError:
                    print("Enter a number.")

        state, reward, done = game.step(state, action)
        print()
        if is_tictactoe and not state.done:
            print(game.render_with_moves(state))
        else:
            print(game.render(state))

    if is_chess:
        if state.winner == 1:
            print("\nModel (White) wins!")
        elif state.winner == -1:
            print("\nYou (Black) win!")
        else:
            print("\nDraw.")
    else:
        if state.winner == 1:
            print("\nModel wins!")
        elif state.winner == -1:
            print("\nYou win!")
        else:
            print("\nDraw!")


def eval_vs_random(game, network, config, device, num_games=100):
    """Evaluate model vs random player."""
    import random as rng
    mcts = MCTS(network, game, config, device)

    results = {1: 0, -1: 0, 0: 0}
    for i in range(num_games):
        state = game.reset()
        while not state.done:
            legal = game.legal_actions(state)
            if state.current_player == 1:
                obs = game.to_tensor(state)
                root = mcts.run(obs, legal, add_noise=False)
                action, _ = select_action(root, temperature=0)
            else:
                action = rng.choice(legal)
            state, _, _ = game.step(state, action)
        results[state.winner] += 1

    print(f"Results vs Random ({num_games} games):")
    print(f"  Wins:   {results[1]}")
    print(f"  Losses: {results[-1]}")
    print(f"  Draws:  {results[0]}")
    print(f"  Win rate: {results[1]/num_games:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MuZero")
    parser.add_argument("--game", type=str, default="tictactoe")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", choices=["play", "eval"], default="eval")
    parser.add_argument("--num-games", type=int, default=100)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    config = get_config(args.game)
    game = get_game(args.game)
    network = load_model(args.checkpoint, game, config, device)

    if args.mode == "play":
        play_interactive(game, network, config, device)
    else:
        eval_vs_random(game, network, config, device, args.num_games)


if __name__ == "__main__":
    main()
