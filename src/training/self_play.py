"""Self-play game generation for MuZero."""

import random

import numpy as np
import torch
from tqdm import tqdm

from ..games.base import Game
from ..mcts.mcts import MCTS, BatchedMCTS, select_action, select_action_gumbel
from .replay_buffer import GameHistory, stack_with_history


def _make_batched_mcts(network, game, config, device):
    """Build the configured batched-MCTS implementation.

    Default: numpy-backed ``BatchedMCTS``. With ``config.use_tensor_mcts``
    set, swap in the GPU tensor-native ``TensorMCTS`` (0 syncs/sim, 1 sync/ply,
    Sampled MuZero only — Gumbel root not yet implemented in the tensor path).
    Both expose the same ``run_batch(observations, legal_actions_list,
    add_noise=True)`` signature returning a list of ``MCTSNode``-equivalents.
    """
    if getattr(config, "use_tensor_mcts", False):
        if getattr(config, "use_gumbel", False):
            raise NotImplementedError(
                "TensorMCTS does not support Gumbel root (use_gumbel=True). "
                "Either disable use_gumbel or use BatchedMCTS."
            )
        from ..mcts.tensor_mcts import TensorMCTS
        dtype_str = getattr(config, "tensor_mcts_hidden_dtype", "float32")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_str not in dtype_map:
            raise ValueError(
                f"Unknown tensor_mcts_hidden_dtype={dtype_str!r}; "
                f"must be one of {list(dtype_map)}."
            )
        return TensorMCTS(
            network, game, config,
            device=device,
            hidden_dtype=dtype_map[dtype_str],
        )
    return BatchedMCTS(network, game, config, device)


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
    """Play a single self-play game using MCTS.

    Under ``config.use_gumbel`` the serial path routes through BatchedMCTS with
    n=1 so Plain Gumbel MuZero's Sequential-Halving root path is reused; action
    selection uses ``select_action_gumbel`` (argmax over sampled + π' target).
    """
    network.eval()
    use_gumbel = bool(getattr(config, "use_gumbel", False))
    mcts_serial = MCTS(network, game, config, device) if not use_gumbel else None
    mcts_batched = BatchedMCTS(network, game, config, device) if use_gumbel else None
    temp_init = get_temperature(training_step, config)

    state = game.reset()
    history = GameHistory(game_name=config.game)
    n_frames = getattr(config, "history_frames", 1)

    move_count = 0
    action_space_size = game.action_space_size
    n_random = getattr(config, "random_opening_plies", 0)
    while not state.done:
        single_frame = game.to_tensor(state)
        legal = game.legal_actions(state)

        if move_count < n_random:
            action = random.choice(legal)
            action_probs = np.zeros(action_space_size, dtype=np.float32)
            action_probs[action] = 1.0
            root_value = 0.0
        else:
            obs = stack_with_history(single_frame, history.observations, n_frames)
            if use_gumbel:
                root = mcts_batched.run_batch([obs], [legal], add_noise=True)[0]
                action, action_probs = select_action_gumbel(root, config, action_space_size)
            else:
                root = mcts_serial.run(obs, legal, add_noise=True)
                temp = temp_init if move_count < config.temperature_drop_step else config.temperature_final
                action, action_probs = select_action(root, temperature=temp)
            root_value = root.value

        history.observations.append(single_frame)
        history.actions.append(action)
        history.policies.append(action_probs)
        history.root_values.append(root_value)
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
    batched_mcts = _make_batched_mcts(network, game, config, device)
    temp_init = get_temperature(training_step, config)
    use_gumbel = bool(getattr(config, "use_gumbel", False))
    action_space_size = game.action_space_size

    states = [game.reset() for _ in range(num_games)]
    histories = [GameHistory(game_name=config.game) for _ in range(num_games)]
    move_counts = [0] * num_games
    active = list(range(num_games))
    n_frames = getattr(config, "history_frames", 1)
    n_random = getattr(config, "random_opening_plies", 0)

    iteration = 0
    log_every = 20  # emit a progress line every 20 lockstep moves (silent for fast games)

    while active:
        # Single-frame current observations (stored 1× per ply, used for sample-time
        # stack reconstruction) and T-frame stacks (passed to MCTS at inference time).
        single_frames = [game.to_tensor(states[g]) for g in active]
        legal_list = [game.legal_actions(states[g]) for g in active]

        # Split active games into random-opening vs MCTS groups.
        mcts_indices = [i for i, g in enumerate(active) if move_counts[g] >= n_random]

        roots_by_active_idx: dict[int, object] = {}
        if mcts_indices:
            obs_list = [
                stack_with_history(single_frames[i], histories[active[i]].observations, n_frames)
                for i in mcts_indices
            ]
            legal_sub = [legal_list[i] for i in mcts_indices]
            mcts_roots = batched_mcts.run_batch(obs_list, legal_sub, add_noise=True)
            for j, i in enumerate(mcts_indices):
                roots_by_active_idx[i] = mcts_roots[j]

        still_active = []
        for i, g in enumerate(active):
            if move_counts[g] < n_random:
                action = random.choice(legal_list[i])
                action_probs = np.zeros(action_space_size, dtype=np.float32)
                action_probs[action] = 1.0
                root_value = 0.0
            else:
                root = roots_by_active_idx[i]
                if use_gumbel:
                    action, action_probs = select_action_gumbel(root, config, action_space_size)
                else:
                    temp = temp_init if move_counts[g] < config.temperature_drop_step else config.temperature_final
                    action, action_probs = select_action(root, temperature=temp)
                root_value = root.value

            # Store the SINGLE-FRAME observation. Sample-time stacking rebuilds
            # the T-frame stack from per-ply observations.
            histories[g].observations.append(single_frames[i])
            histories[g].actions.append(action)
            histories[g].policies.append(action_probs)
            histories[g].root_values.append(root_value)
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


def play_games_parallel_gpu(
    network,
    config,
    num_games: int,
    device: str = "cpu",
    training_step: int = 0,
) -> list[GameHistory]:
    """GPU-resident batched self-play (Phase 5 of the GPU chess plan).

    Same MCTS structure as `play_games_parallel`; the only difference is the
    env. Per-game `to_tensor` / `legal_actions` / `step` calls are replaced
    by single batched calls to `GpuChessGame`. MCTS internals are unchanged
    (latent space — no game state needed past the root).

    Currently chess-specific (only game with a `BatchedGame` impl).
    """
    from ..games.chess import ChessGame
    from ..games.chess_gpu import GpuChessGame

    network.eval()
    chess_game = ChessGame()  # for MCTS action_space + legacy interface bits
    gpu_game = GpuChessGame()
    batched_mcts = _make_batched_mcts(network, chess_game, config, device)
    temp_init = get_temperature(training_step, config)
    use_gumbel = bool(getattr(config, "use_gumbel", False))
    action_space_size = chess_game.action_space_size
    n_frames = getattr(config, "history_frames", 1)

    state = gpu_game.reset_batch(num_games, device=device)
    histories = [GameHistory(game_name=config.game) for _ in range(num_games)]
    move_counts = [0] * num_games
    active = list(range(num_games))
    n_random = getattr(config, "random_opening_plies", 0)

    iteration = 0
    log_every = 20

    while active:
        # One batched call each — replaces N per-game python-chess calls.
        obs_batch = gpu_game.to_tensor_batch(state)        # (N, 19, 8, 8)
        legal_mask_batch = gpu_game.legal_mask(state)      # (N, 4672) bool

        # One bulk transfer to CPU; per-game Python iteration after.
        obs_cpu = obs_batch.cpu()
        legal_mask_cpu = legal_mask_batch.cpu()

        single_frames_active: list[torch.Tensor] = []
        legal_list_active: list[list[int]] = []
        for g in active:
            single_frames_active.append(obs_cpu[g])
            legal_list_active.append(
                legal_mask_cpu[g].nonzero(as_tuple=True)[0].tolist()
            )

        # Only run MCTS for games past the random-opening phase.
        mcts_active_indices = [i for i, g in enumerate(active) if move_counts[g] >= n_random]
        roots_by_active_idx: dict[int, object] = {}
        if mcts_active_indices:
            obs_list_mcts = [
                stack_with_history(single_frames_active[i], histories[active[i]].observations, n_frames)
                for i in mcts_active_indices
            ]
            legal_list_mcts = [legal_list_active[i] for i in mcts_active_indices]
            mcts_roots = batched_mcts.run_batch(obs_list_mcts, legal_list_mcts, add_noise=True)
            for j, i in enumerate(mcts_active_indices):
                roots_by_active_idx[i] = mcts_roots[j]

        actions_per_game = [0] * num_games
        for i, g in enumerate(active):
            if move_counts[g] < n_random:
                action = random.choice(legal_list_active[i])
                action_probs = np.zeros(action_space_size, dtype=np.float32)
                action_probs[action] = 1.0
                root_value = 0.0
            else:
                root = roots_by_active_idx[i]
                if use_gumbel:
                    action, action_probs = select_action_gumbel(root, config, action_space_size)
                else:
                    temp = temp_init if move_counts[g] < config.temperature_drop_step else config.temperature_final
                    action, action_probs = select_action(root, temperature=temp)
                root_value = root.value

            histories[g].observations.append(single_frames_active[i])
            histories[g].actions.append(action)
            histories[g].policies.append(action_probs)
            histories[g].root_values.append(root_value)
            histories[g].legal_actions_list.append(legal_list_active[i])

            actions_per_game[g] = action
            move_counts[g] += 1

        # Step all games (done ones get sentinel action 0 — they're filtered
        # below). step_batch keeps `state.done` sticky so re-stepping a
        # finished game doesn't corrupt its outcome.
        actions_tensor = torch.tensor(actions_per_game, dtype=torch.int64, device=state.device)
        state, rewards, _ = gpu_game.step_batch(state, actions_tensor)
        rewards_cpu = rewards.cpu().tolist()
        done_cpu = state.done.cpu().tolist()
        winner_cpu = state.winner.cpu().tolist()

        still_active: list[int] = []
        terminal_indices: list[int] = []
        for g in active:
            histories[g].rewards.append(rewards_cpu[g])
            if done_cpu[g]:
                histories[g].game_outcome = winner_cpu[g]
                terminal_indices.append(g)
            else:
                still_active.append(g)

        # Append terminal observations (matches play_games_parallel's contract).
        if terminal_indices:
            final_obs = gpu_game.to_tensor_batch(state).cpu()
            for g in terminal_indices:
                histories[g].observations.append(final_obs[g])

        active = still_active
        iteration += 1

        if iteration % log_every == 0:
            done = num_games - len(active)
            done_lengths = [len(h.actions) for h in histories if h.game_outcome is not None]
            avg_done = (sum(done_lengths) / len(done_lengths)) if done_lengths else 0.0
            tqdm.write(
                f"  self-play (gpu) batch: move {iteration}, "
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

    # GPU-resident chess env path. Gated by config.use_gpu_chess (default off).
    # Only kicks in for chess; other games still go through python-chess /
    # python-state implementations.
    use_gpu_chess = (
        bool(getattr(config, "use_gpu_chess", False))
        and config.game == "chess"
        and n_parallel > 1
    )
    if use_gpu_chess:
        histories = []
        remaining = num_games
        iterator = range(0, num_games, n_parallel)
        if show_progress:
            iterator = tqdm(iterator, desc="Self-play (gpu)", leave=False)
        for _ in iterator:
            batch = min(n_parallel, remaining)
            histories.extend(
                play_games_parallel_gpu(network, config, batch, device, training_step)
            )
            remaining -= batch
        return histories

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
