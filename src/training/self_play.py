"""Self-play game generation for MuZero."""

import torch
from tqdm import tqdm

from ..games.base import Game
from ..mcts.mcts import MCTS, BatchedMCTS, select_action
from .replay_buffer import GameHistory


def get_temperature(training_step: int, config) -> float:
    """Return temperature_init for the current training step.

    Applies the temperature_schedule from config: a list of (step_fraction, temperature)
    pairs that progressively lower exploration as training matures.
    """
    temp = config.temperature_init
    schedule = getattr(config, "temperature_schedule", [])
    for frac, t in schedule:
        if training_step >= frac * config.training_steps:
            temp = t
    return temp


def play_game(
    network,
    game: Game,
    config,
    device: str = "cpu",
    training_step: int = 0,
) -> GameHistory:
    """Play a single self-play game using MCTS."""
    network.eval()
    mcts = MCTS(network, game, config, device)
    temp_init = get_temperature(training_step, config)
    use_is = getattr(config, "sample_k", None) is not None

    state = game.reset()
    history = GameHistory(game_name=config.game)

    move_count = 0
    while not state.done:
        obs = game.to_tensor(state)
        legal = game.legal_actions(state)

        root = mcts.run(obs, legal, add_noise=True)

        temp = temp_init if move_count < config.temperature_drop_step else config.temperature_final
        action, action_probs = select_action(root, temperature=temp, use_importance_sampling=use_is)

        history.observations.append(obs)
        history.actions.append(action)
        history.policies.append(action_probs)
        history.root_values.append(root.value)
        history.legal_actions_list.append(legal)

        state, reward, done = game.step(state, action)
        history.rewards.append(reward)
        move_count += 1

    history.game_outcome = state.winner
    history.observations.append(game.to_tensor(state))
    return history


def play_games_parallel(
    network,
    game: Game,
    config,
    num_games: int,
    device: str = "cpu",
    training_step: int = 0,
) -> list[GameHistory]:
    """Run num_games self-play games in parallel using batched MCTS.

    All active games advance one move at a time in lockstep. At each move,
    MCTS simulations across all active games are batched into a single
    network forward pass per simulation step, amortizing GPU kernel overhead.
    """
    network.eval()
    batched_mcts = BatchedMCTS(network, game, config, device)
    temp_init = get_temperature(training_step, config)
    use_is = getattr(config, "sample_k", None) is not None

    states = [game.reset() for _ in range(num_games)]
    histories = [GameHistory(game_name=config.game) for _ in range(num_games)]
    move_counts = [0] * num_games
    active = list(range(num_games))

    iteration = 0
    log_every = 20  # emit a progress line every 20 lockstep moves (silent for fast games)

    while active:
        obs_list = [game.to_tensor(states[g]) for g in active]
        legal_list = [game.legal_actions(states[g]) for g in active]

        roots = batched_mcts.run_batch(obs_list, legal_list, add_noise=True)

        still_active = []
        for i, g in enumerate(active):
            temp = temp_init if move_counts[g] < config.temperature_drop_step else config.temperature_final
            action, action_probs = select_action(roots[i], temperature=temp, use_importance_sampling=use_is)

            histories[g].observations.append(obs_list[i])
            histories[g].actions.append(action)
            histories[g].policies.append(action_probs)
            histories[g].root_values.append(roots[i].value)
            histories[g].legal_actions_list.append(legal_list[i])

            state, reward, _ = game.step(states[g], action)
            histories[g].rewards.append(reward)
            states[g] = state
            move_counts[g] += 1

            if state.done:
                histories[g].game_outcome = state.winner
                histories[g].observations.append(game.to_tensor(state))
            else:
                still_active.append(g)

        active = still_active
        iteration += 1

        if iteration % log_every == 0:
            done = num_games - len(active)
            done_lengths = [len(h.actions) for h in histories if h.game_outcome is not None]
            avg_done = (sum(done_lengths) / len(done_lengths)) if done_lengths else 0.0
            tqdm.write(
                f"  self-play batch: move {iteration}, "
                f"{len(active)}/{num_games} active, "
                f"{done} done"
                + (f" (avg length {avg_done:.0f})" if done_lengths else "")
            )

    return histories


def run_self_play(
    network,
    game: Game,
    config,
    num_games: int,
    device: str = "cpu",
    show_progress: bool = True,
    training_step: int = 0,
) -> list[GameHistory]:
    """Run multiple self-play games, using parallel batched MCTS when configured."""
    n_parallel = getattr(config, "num_parallel_games", 1)

    if n_parallel > 1:
        histories = []
        remaining = num_games
        iterator = range(0, num_games, n_parallel)
        if show_progress:
            iterator = tqdm(iterator, desc="Self-play", leave=False)
        for _ in iterator:
            batch = min(n_parallel, remaining)
            histories.extend(play_games_parallel(network, game, config, batch, device, training_step))
            remaining -= batch
        return histories

    games = []
    iterator = range(num_games)
    if show_progress:
        iterator = tqdm(iterator, desc="Self-play", leave=False)
    for _ in iterator:
        games.append(play_game(network, game, config, device, training_step))
    return games
