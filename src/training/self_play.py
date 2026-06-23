"""Self-play game generation for MuZero."""

import random

import numpy as np
import torch
from tqdm import tqdm

from ..games.base import Game
from ..mcts.mcts import MCTS, BatchedMCTS, select_action, select_action_gumbel
from .replay_buffer import GameHistory, stack_with_history, _sparsify_policy


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
        amp_str = getattr(config, "tensor_mcts_amp_dtype", None)
        amp_dtype = dtype_map[amp_str] if amp_str else None
        return TensorMCTS(
            network, game, config,
            device=device,
            hidden_dtype=dtype_map[dtype_str],
            select_backend=getattr(config, "tensor_mcts_select_backend", "compile"),
            amp_dtype=amp_dtype,
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


def _apply_resignation(histories: list[GameHistory], config) -> list[GameHistory]:
    """Post-hoc consecutive-move resignation (AlphaZero/Leela/KataGo style).

    Engine-agnostic: scans each game's stored STM-relative ``root_values`` and,
    if the side to move's value stays below ``resign_threshold`` for
    ``resign_consecutive`` of ITS OWN moves, that side resigns — the game is
    truncated at that ply and relabeled as a decisive loss for the resigner.

    This protects the decisive LABEL a weak policy would otherwise shuffle into a
    draw (see decisive_signal_plan_2026_06_23.md). It fires off the value head's
    own estimate, so it is inert until the value head can see advantages. It is
    NOT a compute saving (the game was already played out) — label protection
    only. Mutates histories in place and returns the same list.
    """
    if not getattr(config, "resign_enabled", False):
        return histories
    thr = float(getattr(config, "resign_threshold", -0.9))
    need = int(getattr(config, "resign_consecutive", 5))
    if need <= 0:
        return histories

    for h in histories:
        rv = h.root_values
        n_ply = min(len(h.actions), len(rv))
        cnt = [0, 0]                 # consecutive own-move counters: [even, odd]
        resign_ply = None
        for p in range(n_ply):
            side = p & 1
            if rv[p] < thr:
                cnt[side] += 1
                if cnt[side] >= need:
                    resign_ply = p
                    break
            else:
                cnt[side] = 0
        if resign_ply is None or resign_ply < 1:
            continue
        p = resign_ply
        # Side to move at ply p resigns (even ply = white = player 1). Outcome is
        # player-1 POV: white resigns → black wins → -1; black resigns → +1.
        h.game_outcome = -1.0 if (p & 1) == 0 else 1.0
        h.draw_by_repetition = False
        h.draw_by_no_progress = False
        h.resigned = True
        # Truncate: keep p played moves and p+1 observations (obs[p] = the resign
        # position, now terminal). Preserves len(obs) == len(actions) + 1.
        h.actions = h.actions[:p]
        h.policies = h.policies[:p]
        h.root_values = h.root_values[:p]
        h.rewards = h.rewards[:p]
        h.legal_actions_list = h.legal_actions_list[:p]
        h.observations = h.observations[:p + 1]
    return histories


def get_material_value_weight(training_step: int, config) -> float:
    """Annealed material/score-margin blend weight for the current step.

    Linearly decays ``material_value_weight`` → ``material_value_weight_final``
    over [0, ``material_value_anneal_frac`` · training_steps]. With
    anneal_frac == 0 (or init weight 0) the weight is constant — back-compat.
    Strong material shaping early to escape the flat-target draw basin, then
    fade so the true game outcome dominates (see decisive_signal_plan_2026_06_23.md).
    """
    w_init = float(getattr(config, "material_value_weight", 0.0))
    frac = float(getattr(config, "material_value_anneal_frac", 0.0))
    if frac <= 0.0 or w_init == 0.0:
        return w_init
    w_final = float(getattr(config, "material_value_weight_final", 0.0))
    total = max(1, int(getattr(config, "training_steps", 1)))
    t = min(1.0, max(0.0, training_step / (frac * total)))
    return w_init + (w_final - w_init) * t


def get_material_head_weight(training_step: int, config) -> float:
    """Annealed aux-loss weight for the material-margin head. Shares the
    ``material_value_anneal_frac`` timeline with the value-target blend so the
    material influence fades in lockstep. frac=0 ⇒ constant at the init weight."""
    w_init = float(getattr(config, "material_head_loss_weight", 0.0))
    frac = float(getattr(config, "material_value_anneal_frac", 0.0))
    if frac <= 0.0 or w_init == 0.0:
        return w_init
    w_final = float(getattr(config, "material_head_loss_weight_final", 0.0))
    total = max(1, int(getattr(config, "training_steps", 1)))
    t = min(1.0, max(0.0, training_step / (frac * total)))
    return w_init + (w_final - w_init) * t


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
        history.policies.append(_sparsify_policy(action_probs))
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
            histories[g].policies.append(_sparsify_policy(action_probs))
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
    gpu_game.max_plies = int(getattr(config, "max_plies", gpu_game.max_plies))
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

    # legal_mask for ply 0 is computed once here; subsequent plies reuse the
    # mask returned by step_batch_with_legal (it computed the new state's
    # legal_mask internally for terminal detection — saves ~16 ms/ply).
    legal_mask_batch = gpu_game.legal_mask(state)

    while active:
        # One batched call each — replaces N per-game python-chess calls.
        obs_batch = gpu_game.to_tensor_batch(state)        # (N, 19, 8, 8)

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
            histories[g].policies.append(_sparsify_policy(action_probs))
            histories[g].root_values.append(root_value)
            histories[g].legal_actions_list.append(legal_list_active[i])

            actions_per_game[g] = action
            move_counts[g] += 1

        # Step all games (done ones get sentinel action 0 — they're filtered
        # below). step_batch keeps `state.done` sticky so re-stepping a
        # finished game doesn't corrupt its outcome. Also returns the new
        # state's legal_mask (carried into the next iteration).
        actions_tensor = torch.tensor(actions_per_game, dtype=torch.int64, device=state.device)
        state, rewards, _, legal_mask_batch = gpu_game.step_batch_with_legal(state, actions_tensor)
        rewards_cpu = rewards.cpu().tolist()
        done_cpu = state.done.cpu().tolist()
        winner_cpu = state.winner.cpu().tolist()
        threefold_cpu = (
            state.terminal_threefold.cpu().tolist()
            if state.terminal_threefold is not None
            else [False] * len(done_cpu)
        )
        no_progress_cpu = (
            state.terminal_no_progress.cpu().tolist()
            if state.terminal_no_progress is not None
            else [False] * len(done_cpu)
        )

        still_active: list[int] = []
        terminal_indices: list[int] = []
        for g in active:
            histories[g].rewards.append(rewards_cpu[g])
            if done_cpu[g]:
                histories[g].game_outcome = winner_cpu[g]
                histories[g].draw_by_repetition = bool(threefold_cpu[g])
                histories[g].draw_by_no_progress = bool(no_progress_cpu[g])
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


def play_games_parallel_gpu_resident(
    network,
    config,
    num_games: int,
    device: str = "cuda",
    training_step: int = 0,
) -> list[GameHistory]:
    """Fully GPU-resident batched self-play.

    All per-ply state lives on the GPU: observations, legal masks, MCTS
    output (action_probs, root_value), env state, sticky-done mask, game
    outcome, per-game move count. The CPU sees the data exactly once at
    end-of-batch, when we transfer the accumulated tensors and build the
    Python ``GameHistory`` records the rest of the pipeline expects.

    Sync count per ply: **0**. Sync count per batch: ~10 (the end-of-batch
    bulk transfer of stacked history + outcome + length).

    Requires ``config.use_tensor_mcts=True`` (only TensorMCTS exposes the
    GPU-resident ``run_batch_gpu`` entry point) and the chess GPU env.
    Random opening plies, per-game temperature schedule, and Sampled MuZero
    are all handled GPU-side.
    """
    from ..games.chess import ChessGame
    from ..games.chess_gpu import GpuChessGame
    from ..mcts.tensor_mcts import TensorMCTS, select_action_gpu

    if not getattr(config, "use_tensor_mcts", False):
        raise ValueError(
            "play_games_parallel_gpu_resident requires use_tensor_mcts=True."
        )
    if getattr(config, "use_gumbel", False):
        raise NotImplementedError(
            "Gumbel root not supported in the GPU-resident path."
        )

    network.eval()
    chess_game = ChessGame()
    gpu_game = GpuChessGame()
    gpu_game.max_plies = int(getattr(config, "max_plies", gpu_game.max_plies))

    dtype_str = getattr(config, "tensor_mcts_hidden_dtype", "float32")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    hidden_dtype = dtype_map[dtype_str]
    amp_str = getattr(config, "tensor_mcts_amp_dtype", None)
    amp_dtype = dtype_map[amp_str] if amp_str else None
    mcts = TensorMCTS(
        network, chess_game, config, device=device, hidden_dtype=hidden_dtype,
        select_backend=getattr(config, "tensor_mcts_select_backend", "compile"),
        use_subtree_reuse=getattr(config, "tensor_mcts_subtree_reuse", False),
        amp_dtype=amp_dtype,
    )

    action_space_size = chess_game.action_space_size
    n_frames = int(getattr(config, "history_frames", 1))
    n_random = int(getattr(config, "random_opening_plies", 0))
    temp_init = get_temperature(training_step, config)
    temp_final = float(config.temperature_final)
    temp_drop = int(config.temperature_drop_step)

    # Cap loop length so we always have a static upper bound; ChessGame.max_plies
    # does the in-engine termination, alive_mask handles per-game stopping.
    max_plies_cap = int(getattr(config, "max_plies", getattr(ChessGame, "max_plies", 400)))

    state = gpu_game.reset_batch(num_games, device=device)

    # Single-frame observation history for AlphaZero-style stacking. Newest
    # frame at slot 0; rolled forward each ply. Initial all-zero (matches
    # stack_with_history's "missing past frames are zero-padded" behavior).
    sample_obs = gpu_game.to_tensor_batch(state)                   # [N, C, H, W]
    c, h, w = sample_obs.shape[1:]
    obs_window = torch.zeros(
        num_games, n_frames, c, h, w, device=device, dtype=sample_obs.dtype
    )
    # Pre-compute legal_mask once for the initial state. Subsequent plies
    # reuse the legal_mask returned by ``step_batch_with_legal`` instead of
    # recomputing — saves ~16 ms/ply (a duplicate _legal_mask_impl call).
    next_legal_mask = gpu_game.legal_mask(state)
    # Pre-cache temperature tensors (avoid per-ply tensor construction).
    temp_init_tensor = torch.full((num_games,), float(temp_init), dtype=torch.float32, device=device)
    temp_final_tensor = torch.full((num_games,), float(temp_final), dtype=torch.float32, device=device)

    # Per-batch GPU history accumulators. Stack at end; one transfer.
    obs_per_ply: list[torch.Tensor] = []           # [T] of [N, C, H, W]
    actions_per_ply: list[torch.Tensor] = []       # [T] of [N] int64
    policies_per_ply: list[torch.Tensor] = []      # [T] of [N, A] float32
    values_per_ply: list[torch.Tensor] = []        # [T] of [N] float32
    rewards_per_ply: list[torch.Tensor] = []       # [T] of [N] float32
    legal_masks_per_ply: list[torch.Tensor] = []   # [T] of [N, A] bool

    alive_mask = torch.ones(num_games, dtype=torch.bool, device=device)
    game_outcome = torch.zeros(num_games, dtype=torch.int32, device=device)
    terminal_threefold = torch.zeros(num_games, dtype=torch.bool, device=device)
    terminal_no_progress = torch.zeros(num_games, dtype=torch.bool, device=device)
    game_length = torch.zeros(num_games, dtype=torch.int32, device=device)
    move_count = torch.zeros(num_games, dtype=torch.int32, device=device)

    iteration = 0
    log_every = 20

    # Pre-allocate the all-False sentinel mask for terminal-row safety.
    # For terminal games, legal_mask is all-zero (no legal moves) and
    # softmax/multinomial would NaN out. We mark action 0 as legal so MCTS
    # picks something; step_batch keeps state.done sticky so the choice is
    # discarded by the alive_mask gating below.
    sentinel_legal = torch.zeros(num_games, action_space_size, dtype=torch.bool, device=device)
    sentinel_legal[:, 0] = True

    for ply in range(max_plies_cap):
        # 1. Build batched obs GPU-side. The legal_mask was already computed
        #    by the previous ply's step_batch_with_legal (or once up-front
        #    for ply 0) — ~16 ms/ply saved vs recomputing.
        single_obs = gpu_game.to_tensor_batch(state)                  # [N, C, H, W]
        legal_mask = next_legal_mask                                   # carried over
        # Sentinel any all-zero rows (terminal states) so MCTS multinomial
        # doesn't NaN.
        any_legal = legal_mask.any(dim=1, keepdim=True)               # [N, 1] bool
        legal_mask = torch.where(any_legal, legal_mask, sentinel_legal)

        # 2. Update rolling history window (newest frame at slot 0).
        obs_window = torch.roll(obs_window, shifts=1, dims=1)
        obs_window[:, 0] = single_obs
        # Stack along channel dim → [N, n_frames * C, H, W] (matches stack_with_history).
        stacked_obs = obs_window.reshape(num_games, n_frames * c, h, w)

        # 3. MCTS — but skip for the random-opening plies (all alive games
        # have move_count == ply during the opening because move_count
        # increments by alive_mask each ply and starts at 0). For ply >=
        # n_random, every alive game is out of opening.
        if ply < n_random:
            # Pure-random opening: uniform sample from legal_mask, no MCTS.
            uniform_logits = legal_mask.to(torch.float32)
            action = torch.multinomial(uniform_logits, 1).squeeze(1).long()
            policy = torch.zeros(
                num_games, action_space_size, device=device, dtype=torch.float32
            )
            policy.scatter_(1, action.unsqueeze(1), 1.0)
            value = torch.zeros(num_games, device=device, dtype=torch.float32)
        else:
            root_data = mcts.run_batch_gpu(stacked_obs, legal_mask, add_noise=True)

            # 4. Per-game temperature for sampling (AlphaZero schedule).
            #    Picks from pre-cached tensors to avoid per-ply construction.
            is_post_drop = move_count >= temp_drop
            temperature = torch.where(is_post_drop, temp_final_tensor, temp_init_tensor)
            action, policy = select_action_gpu(
                root_data["child_actions"],
                root_data["child_visits"],
                temperature,
                action_space_size,
            )
            value = root_data["root_value"]

        # 6. Step env. ``step_batch_with_legal`` returns the new state's
        #    legal_mask alongside (already computed inside for terminal
        #    detection); reuse on next ply rather than recomputing.
        state, rewards, _, next_legal_mask = gpu_game.step_batch_with_legal(state, action)

        # 7. Append to GPU accumulators (no transfer).
        obs_per_ply.append(single_obs)
        actions_per_ply.append(action)
        policies_per_ply.append(policy)
        values_per_ply.append(value)
        rewards_per_ply.append(rewards.to(torch.float32))
        legal_masks_per_ply.append(legal_mask)

        # 8. Subtree reuse: advance the MCTS root to the chosen action's
        # subtree, in preparation for the next ply. Only when the prior
        # search actually ran (skip during random openings — those don't
        # populate a tree).
        if getattr(config, "tensor_mcts_subtree_reuse", False) and ply >= n_random:
            mcts.advance_root(action, legal_mask)

        # 9. Update accounting (sticky-done semantics).
        newly_done = state.done & alive_mask
        game_outcome = torch.where(newly_done, state.winner.to(torch.int32), game_outcome)
        if state.terminal_threefold is not None:
            terminal_threefold = torch.where(
                newly_done, state.terminal_threefold, terminal_threefold
            )
        if state.terminal_no_progress is not None:
            terminal_no_progress = torch.where(
                newly_done, state.terminal_no_progress, terminal_no_progress
            )
        game_length = torch.where(alive_mask, game_length + 1, game_length)
        move_count = move_count + alive_mask.to(torch.int32)
        alive_mask = alive_mask & ~state.done

        iteration += 1
        if iteration % log_every == 0:
            # Best-effort progress log — one sync per 20 plies (negligible).
            n_done = int((~alive_mask).sum().item())
            tqdm.write(
                f"  self-play (gpu-resident) batch: move {iteration}, "
                f"{num_games - n_done}/{num_games} active, {n_done} done"
            )

        # Optional early-exit (one sync per ply if checked every iter).
        # Amortize by checking every 16 plies — much cheaper than per-ply sync.
        if (ply & 15) == 15 and not bool(alive_mask.any()):
            break

    # 9. Capture final terminal observation.
    final_obs = gpu_game.to_tensor_batch(state)                      # [N, C, H, W]

    # 10. Stack and bulk-transfer to CPU. ONE pass.
    obs_stack = torch.stack(obs_per_ply, dim=1)                       # [N, T, C, H, W]
    actions_stack = torch.stack(actions_per_ply, dim=1)               # [N, T]
    policies_stack = torch.stack(policies_per_ply, dim=1)             # [N, T, A]
    values_stack = torch.stack(values_per_ply, dim=1)                 # [N, T]
    rewards_stack = torch.stack(rewards_per_ply, dim=1)               # [N, T]
    legal_masks_stack = torch.stack(legal_masks_per_ply, dim=1)       # [N, T, A]

    obs_cpu = obs_stack.cpu().numpy()
    actions_cpu = actions_stack.cpu().numpy()
    policies_cpu = policies_stack.cpu().numpy()
    values_cpu = values_stack.cpu().numpy()
    rewards_cpu = rewards_stack.cpu().numpy()
    legal_masks_cpu = legal_masks_stack.cpu().numpy()
    final_obs_cpu = final_obs.cpu().numpy()
    game_length_cpu = game_length.cpu().numpy()
    game_outcome_cpu = game_outcome.cpu().numpy()
    terminal_threefold_cpu = terminal_threefold.cpu().numpy()
    terminal_no_progress_cpu = terminal_no_progress.cpu().numpy()

    # 11. Build GameHistory objects.
    histories: list[GameHistory] = []
    for g in range(num_games):
        L = int(game_length_cpu[g])
        h_g = GameHistory(game_name=config.game)
        for t in range(L):
            # .copy() / torch.from_numpy(...).clone() materializes per-ply slices.
            # Without this, each appended view pins the entire [N, T, ...] parent
            # array via numpy's `.base` reference — every GameHistory in the buffer
            # held the whole batch's policies (~1.9 GB) and obs (~0.5 GB) parents
            # alive, leaking ~1.5 GB per self-play batch.
            h_g.observations.append(torch.from_numpy(obs_cpu[g, t]).clone())
            h_g.actions.append(int(actions_cpu[g, t]))
            h_g.policies.append(_sparsify_policy(policies_cpu[g, t]))
            h_g.root_values.append(float(values_cpu[g, t]))
            legal_idx = legal_masks_cpu[g, t].nonzero()[0].tolist()
            h_g.legal_actions_list.append(legal_idx)
            h_g.rewards.append(float(rewards_cpu[g, t]))
        h_g.game_outcome = int(game_outcome_cpu[g])
        h_g.draw_by_repetition = bool(terminal_threefold_cpu[g])
        h_g.draw_by_no_progress = bool(terminal_no_progress_cpu[g])
        h_g.observations.append(torch.from_numpy(final_obs_cpu[g]).clone())
        histories.append(h_g)

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
        # Pick fully GPU-resident loop (zero per-ply syncs) when both
        # use_tensor_mcts and use_gpu_resident_self_play are set.
        use_resident = (
            bool(getattr(config, "use_gpu_resident_self_play", False))
            and bool(getattr(config, "use_tensor_mcts", False))
        )
        play_fn = (
            play_games_parallel_gpu_resident if use_resident else play_games_parallel_gpu
        )
        desc = "Self-play (gpu-resident)" if use_resident else "Self-play (gpu)"

        histories = []
        remaining = num_games
        iterator = range(0, num_games, n_parallel)
        if show_progress:
            iterator = tqdm(iterator, desc=desc, leave=False)
        for _ in iterator:
            batch = min(n_parallel, remaining)
            histories.extend(
                play_fn(network, config, batch, device, training_step)
            )
            remaining -= batch
        return _apply_resignation(histories, config)

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
        return _apply_resignation(histories, config)

    games = []
    iterator = range(num_games)
    if show_progress:
        iterator = tqdm(iterator, desc="Self-play", leave=False)
    for _ in iterator:
        games.append(play_game(network, game, config, device, training_step))
    return _apply_resignation(games, config)
