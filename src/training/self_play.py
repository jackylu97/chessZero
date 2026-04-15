"""Self-play game generation for MuZero."""

import torch
from tqdm import tqdm

from ..games.base import Game
from ..mcts.mcts import MCTS, select_action
from .replay_buffer import GameHistory


def play_game(
    network,
    game: Game,
    config,
    device: str = "cpu",
) -> GameHistory:
    """Play a single self-play game using MCTS.

    Returns a GameHistory with full trajectory data.
    """
    network.eval()
    mcts = MCTS(network, game, config, device)

    state = game.reset()
    history = GameHistory(game_name=config.game)

    move_count = 0
    while not state.done:
        obs = game.to_tensor(state)
        legal = game.legal_actions(state)

        # Run MCTS
        root = mcts.run(obs, legal, add_noise=True)

        # Select action with temperature
        temp = config.temperature_init if move_count < config.temperature_drop_step else config.temperature_final
        action, action_probs = select_action(root, temperature=temp)

        # Store trajectory data
        history.observations.append(obs)
        history.actions.append(action)
        history.policies.append(action_probs)
        history.root_values.append(root.value)

        # Step game
        state, reward, done = game.step(state, action)
        history.rewards.append(reward)

        move_count += 1

    # Set game outcome from player 1's perspective
    history.game_outcome = state.winner  # 1, -1, or 0
    # Last observation (terminal state)
    history.observations.append(game.to_tensor(state))

    return history


def run_self_play(
    network,
    game: Game,
    config,
    num_games: int,
    device: str = "cpu",
    show_progress: bool = True,
) -> list[GameHistory]:
    """Run multiple self-play games.

    Returns list of GameHistory objects.
    """
    games = []
    iterator = range(num_games)
    if show_progress:
        iterator = tqdm(iterator, desc="Self-play", leave=False)

    for _ in iterator:
        history = play_game(network, game, config, device)
        games.append(history)

    return games
