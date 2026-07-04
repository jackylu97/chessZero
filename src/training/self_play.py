"""Self-play game generation for MuZero."""

import random
import time

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
    # AlphaZero-style calibration holdout: this fraction of would-resign games is
    # played to its natural end instead of resigned, so the false-positive rate
    # (value head said "lost" but the side did NOT actually lose) is measurable.
    holdout_frac = min(max(float(getattr(config, "resign_holdout_frac", 0.0)), 0.0), 1.0)

    exempt_seeded = bool(getattr(config, "resign_exempt_seeded", False))
    for h in histories:
        if getattr(h, "tb_filled", False):
            # TB-rollout-filled games already carry the true (tablebase) decisive
            # outcome and end in an actual mate demonstration — never re-truncate.
            continue
        if exempt_seeded and h.start_fen is not None:
            # Seeded games: value labels are TB-true regardless (all plies in-TB),
            # and resignation would truncate exactly the conversion-practice tails
            # seeding exists to generate. See config.resign_exempt_seeded.
            continue
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
        # Which COLOR is to move at ply p? Normal games start white-to-move (ply 0 =
        # white), but endgame-seeded games can start from a BLACK-to-move FEN — so derive
        # the start side from start_fen rather than assuming ply 0 == white. Without this,
        # black-start seeded games get their resignation OUTCOME LABEL inverted (game_outcome
        # feeds the value target) and their correct resignations miscounted as false
        # positives. FEN field [1] is 'w'/'b'. The cnt[] parity counters above are
        # color-agnostic (each side owns one ply parity), so only the color attribution here
        # needs the start side.
        sf = getattr(h, "start_fen", None)
        start_white = (sf.split()[1] == "w") if sf else True
        white_resigns = (start_white == (p % 2 == 0))
        # Calibration holdout: leave this game un-resigned (natural outcome) and
        # record whether resignation WOULD have been a false positive — i.e. the
        # would-have-resigned side did NOT actually lose. resign_false_positive
        # over the holdout sample is AlphaZero's <5% threshold-calibration metric.
        if holdout_frac > 0.0 and random.random() < holdout_frac:
            h.resign_holdout = True
            actually_lost = (h.game_outcome == -1.0) if white_resigns else (h.game_outcome == 1.0)
            h.resign_false_positive = not actually_lost
            continue
        # Side to move at ply p resigns (even ply = white = player 1). Outcome is
        # player-1 POV: white resigns → black wins → -1; black resigns → +1.
        h.game_outcome = -1.0 if white_resigns else 1.0
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


def _apply_tb_rollout_fill(histories: list[GameHistory], prober_params: dict) -> int:
    """Win adjudication by demonstration (tb_rollout_fill, strategy_2026_07_02.md).

    For each NON-seeded game whose deferred-relabel ``tablebase_values`` reveal a
    decisive in-TB ply whose verdict CONTRADICTS the played outcome (the classic
    won-but-shuffled-to-a-draw game): truncate the history at the first such ply
    and finish the game with TB-optimal play by BOTH sides (``tb_playout``).

    Two effects the per-ply value relabel cannot deliver:
    (1) ``game_outcome`` becomes the TRUE result for the ENTIRE trajectory — the
        pre-TB middlegame plies get a decisive z instead of the V^π draw poison.
    (2) The appended tail is an on-distribution conversion demonstration
        (soft win-preserving DTZ policy targets at states the model itself
        reached) — no search-side steering, no covariate shift.

    Seeded games are exempt: they START in a decisive TB position, so filling
    would replace the model's own practice data at ply 0 (the anchor archive
    already supplies pure demonstrations). Mutates histories in place; returns
    the number of games filled.
    """
    from ..games.chess import ChessGame
    from ..games.syzygy_probe import _get_local_prober
    from .tb_playout import tb_playout, TBPlayoutError

    prober = _get_local_prober(prober_params)
    game = ChessGame()
    nan = float("nan")
    n_filled = 0
    for h in histories:
        if h.start_fen is not None or not h.tablebase_values:
            continue
        # First decisive in-TB ply. Win magnitudes are >= 1 - value_dtz_shape
        # (>= 0.5 at the production shape 0.5); draws are exactly 0.
        fill_idx = None
        for i, tv in enumerate(h.tablebase_values):
            if tv == tv and abs(tv) >= 0.45:
                fill_idx = i
                break
        if fill_idx is None or fill_idx >= len(h.actions):
            continue
        # TB verdict in white POV (non-seeded games start white-to-move, so ply
        # parity gives the side to move). Skip when the played outcome already
        # matches — the model's own conversion is on-policy gold.
        stm_white = (fill_idx % 2 == 0)
        verdict = 1.0 if ((h.tablebase_values[fill_idx] > 0) == stm_white) else -1.0
        if float(h.game_outcome) == verdict:
            continue
        # Reconstruct the board at fill_idx by replaying the stored actions
        # (same replay path buffer serialization uses — proven action encoding).
        state = game.reset()
        for a in h.actions[:fill_idx]:
            state, _, _ = game.step(state, a)
        try:
            res = tb_playout(state.board.copy(), prober)
        except TBPlayoutError:
            continue  # missing table / 50-move risk — leave the game untouched
        # Truncate every per-ply list at fill_idx (obs keeps the fill-start
        # position at index fill_idx: len(obs) == len(actions) + 1 invariant).
        L_old = len(h.actions)
        h.actions = h.actions[:fill_idx]
        h.policies = h.policies[:fill_idx]
        h.root_values = h.root_values[:fill_idx]
        h.rewards = h.rewards[:fill_idx]
        h.legal_actions_list = h.legal_actions_list[:fill_idx]
        h.observations = h.observations[:fill_idx + 1]
        tbv = h.tablebase_values[:fill_idx]
        tbm = (h.tablebase_moves_left[:fill_idx] if h.tablebase_moves_left
               else [nan] * fill_idx)
        tbm += [nan] * (fill_idx - len(tbm))
        tbp = (h.tablebase_policy[:fill_idx] if h.tablebase_policy
               else [None] * fill_idx)
        tbp += [None] * (fill_idx - len(tbp))
        # Append the demonstration tail, stepping ChessGame for obs + legals.
        for k, a in enumerate(res["actions"]):
            h.legal_actions_list.append(game.legal_actions(state))
            h.actions.append(a)
            h.policies.append(res["policies"][k])
            h.root_values.append(res["root_values"][k])
            h.rewards.append(res["rewards"][k])
            state, _, _ = game.step(state, a)
            h.observations.append(game.to_tensor(state))
        h.tablebase_values = tbv + res["tablebase_values"]
        h.tablebase_moves_left = tbm + res["tablebase_moves_left"]
        h.tablebase_policy = tbp + res["tablebase_policy"]
        h.game_outcome = verdict
        h.draw_by_repetition = False
        h.draw_by_no_progress = False
        h.tb_filled = True
        h.tb_authored = True   # fill-tail policies are TB demos — shield from reanalyze
        n_filled += 1
    return n_filled


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


def get_warmstart_sample_frac(training_step: int, config) -> float:
    """Annealed warmstart-anchor sampling fraction for the current step.

    Linearly decays ``warmstart_sample_frac`` → ``warmstart_sample_frac_final``
    over [0, ``warmstart_anneal_frac`` · training_steps]. With anneal_frac == 0
    the fraction is constant — back-compat. The Stockfish anchor calibrates the
    value head early (anti V^π≈draw collapse), but its middlegame distribution
    diverges from the model's own play and dilutes self-play conversion learning;
    as self-play accrues its own decisive signal (TB conversions) the anchor's
    cost outweighs its benefit, so fade it. See decisive_signal_plan_2026_06_23.md.
    """
    f_init = float(getattr(config, "warmstart_sample_frac", 0.0))
    frac = float(getattr(config, "warmstart_anneal_frac", 0.0))
    if frac <= 0.0 or f_init == 0.0:
        return f_init
    f_final = float(getattr(config, "warmstart_sample_frac_final", 0.0))
    total = max(1, int(getattr(config, "training_steps", 1)))
    t = min(1.0, max(0.0, training_step / (frac * total)))
    return f_init + (f_final - f_init) * t


def get_tb_policy_weight(training_step: int, config) -> float:
    """Annealed soft-TB-policy-relabel weight. Linearly decays ``tb_policy_weight`` →
    ``tb_policy_weight_final`` over [0, ``tb_policy_anneal_frac`` · training_steps]; with
    anneal_frac == 0 it's constant. The TB policy boost is a teacher that should FADE so
    the model internalizes conversion rather than leaning on it (probe-is-a-crutch)."""
    w_init = float(getattr(config, "tb_policy_weight", 0.0))
    frac = float(getattr(config, "tb_policy_anneal_frac", 0.0))
    if frac <= 0.0 or w_init == 0.0:
        return w_init
    w_final = float(getattr(config, "tb_policy_weight_final", 0.0))
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
    start_fens: list[str] | None = None,
) -> list[GameHistory]:
    """``start_fens`` (len == num_games): seed each game from a FEN (endgame
    curriculum) instead of the standard start. Stays fully GPU-resident — only the
    initial batched state changes (via from_python_chess). Random opening is
    disabled for seeded games (it would corrupt the endgame); start_fen is stamped
    on each GameHistory so buffer reconstruction replays from the seed."""
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
    # Gumbel root IS supported in the GPU-resident path since 2026-07-04
    # (TensorMCTS._initialize_root_gumbel + Sequential Halving root override;
    # oracle-parity-tested vs the numpy BatchedMCTS in test_gumbel_tensor_mcts).
    # run_batch_gpu returns gumbel_action/gumbel_policy, consumed below in
    # place of select_action_gpu + the temperature schedule.

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
    select_backend = getattr(config, "tensor_mcts_select_backend", "compile")
    if getattr(config, "use_gumbel", False) and select_backend == "triton":
        # The Triton PUCT-walk kernel has no Sequential-Halving root-override
        # path; fall back to the compiled selector (same math, ~10-20% slower
        # select step — small vs the per-sim network forward).
        tqdm.write("  gumbel: select_backend triton -> compile (root override unsupported in kernel)")
        select_backend = "compile"
    mcts = TensorMCTS(
        network, chess_game, config, device=device, hidden_dtype=hidden_dtype,
        select_backend=select_backend,
        use_subtree_reuse=getattr(config, "tensor_mcts_subtree_reuse", False),
        amp_dtype=amp_dtype,
    )

    action_space_size = chess_game.action_space_size
    n_frames = int(getattr(config, "history_frames", 1))
    seeded = start_fens is not None
    # Per-game opening plan. Seeded games (a start FEN) never open randomly — a
    # random opening would wreck the position. start_fens may contain None
    # entries (merged_seed_batch: normal + seeded games in ONE sweep). For
    # NORMAL games, two opening modes:
    #  - legacy (default): random_opening_plies uniform-random plies, ONE-HOT
    #    policy targets (unchanged behavior);
    #  - ε-mixture (opening_mix_mean_plies > 0): per-game r ~ U[1, 2·mean]
    #    plies, uniform-random w.p. opening_uniform_frac (model-independent
    #    diversity floor) else raw-policy softmax at opening_policy_temp
    #    (KataGo initGamesWithPolicy). Opening plies store ZERO policy targets
    #    (make_target's zero-CE no-op — no gradient toward opening moves).
    fen_list = list(start_fens) if start_fens is not None else [None] * num_games
    assert len(fen_list) == num_games, "start_fens must have length num_games"
    is_seed_l = [f is not None for f in fen_list]
    mix_mean = int(getattr(config, "opening_mix_mean_plies", 0))
    mix_on = mix_mean > 0
    n_random_scalar = int(getattr(config, "random_opening_plies", 0))
    open_uniform_frac = float(getattr(config, "opening_uniform_frac", 0.15))
    open_policy_temp = max(float(getattr(config, "opening_policy_temp", 1.5)), 1e-3)
    open_len_l, open_uniform_l = [], []
    for sd in is_seed_l:
        if sd:
            open_len_l.append(0); open_uniform_l.append(False)
        elif mix_on:
            open_len_l.append(random.randint(1, 2 * mix_mean))
            open_uniform_l.append(random.random() < open_uniform_frac)
        else:
            open_len_l.append(n_random_scalar); open_uniform_l.append(True)
    open_len_t = torch.tensor(open_len_l, dtype=torch.int32, device=device)
    open_uniform_t = torch.tensor(open_uniform_l, dtype=torch.bool, device=device)
    max_open = max(open_len_l) if open_len_l else 0
    min_open = min(open_len_l) if open_len_l else 0
    # Downstream batch-wide gates (subtree reuse, inline-prober skip) key on the
    # LAST possible opening ply; keep the legacy name.
    n_random = max_open
    temp_init = get_temperature(training_step, config)
    temp_final = float(config.temperature_final)
    temp_drop = int(config.temperature_drop_step)
    # Root repetition-draw penalty: steer the winning side away from shuffling
    # into a draw. Detect repeating root moves from the live board and pin them
    # to draw_score in the search (TensorMCTS root-terminal-draws). Off by default.
    use_terminal_draws = bool(getattr(config, "root_terminal_draws", False))
    term_min_repeats = int(getattr(config, "root_terminal_draws_min_repeats", 2))
    term_include_stalemate = bool(getattr(config, "root_terminal_draws_include_stalemate", True))

    # Root Syzygy probing: at each ply, classify the root's legal moves against
    # tablebases (≤ tb_max_pieces) and overwrite the root children's value_score
    # in the search toward the DTZ-optimal conversion move. Off by default.
    use_tb_root = bool(getattr(config, "tb_root_probe", False))
    tb_prober = None
    if use_tb_root:
        from ..games.syzygy_probe import (
            SyzygyRootProber, state_to_board as _state_to_board,
            relabel_fens as _relabel_fens, _prober_params,
        )
        tb_prober = SyzygyRootProber(
            getattr(config, "tb_path", "data/syzygy"),
            max_pieces=int(getattr(config, "tb_max_pieces", 5)),
            dtz_weight=float(getattr(config, "tb_dtz_weight", 0.05)),
            draw_score=float(getattr(config, "draw_score", 0.0)),
            value_dtz_shape=float(getattr(config, "tb_value_dtz_shape", 0.5)),
            gaviota_path=getattr(config, "tb_gaviota_path", None),
            policy_win_thresh=float(getattr(config, "tb_policy_win_thresh", 0.5)),
            policy_temp=float(getattr(config, "tb_policy_temp", 0.3)),
        )
    # Build the soft TB policy target when the policy relabel is on (needs per-move
    # _classify). Independent of search-side steering (tb_steer_policy).
    tb_policy_relabel = bool(float(getattr(config, "tb_policy_weight", 0.0)) > 0.0)
    # Search-side steering: when ON, the prober's per-move values BIAS the search, so
    # it MUST run inline (the original path). When OFF (Lc0-faithful default) the
    # prober only produces TRAINING targets (value/DTM/policy) — pure functions of each
    # ply's board — so we DEFER them to one batched, process-pooled pass after the game
    # batch concludes (removes ~56% of endgame self-play wall from the GPU hot path).
    steer_enabled = bool(getattr(config, "tb_steer_policy", False))
    need_per_move = steer_enabled or tb_policy_relabel
    defer_relabel = (tb_prober is not None) and (not steer_enabled)
    relabel_workers = int(getattr(config, "tb_relabel_workers", 0))
    # FENs of the pre-step root board at each ply (deferred path only); None where the
    # game is out of TB range / in the random opening. [T] list of [N] str-or-None.
    tb_fen_per_ply: list = []

    # Cap loop length so we always have a static upper bound; ChessGame.max_plies
    # does the in-engine termination, alive_mask handles per-game stopping.
    max_plies_cap = int(getattr(config, "max_plies", getattr(ChessGame, "max_plies", 400)))

    if any(is_seed_l):
        # Any seeded game present: build the batch from explicit boards. None
        # entries (merged_seed_batch normal games) get the standard start —
        # identical state to reset_batch, just via the python-chess bridge.
        import chess as _chess
        state = gpu_game.from_python_chess(
            [_chess.Board(f) if f is not None else _chess.Board() for f in fen_list],
            device=device)
    else:
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
    tb_values_per_ply: list[torch.Tensor] = []     # [T] of [N] float32 (NaN if not in TB)
    tb_moves_left_per_ply: list[torch.Tensor] = [] # [T] of [N] float32 |DTM| (NaN if draw/not in TB)
    tb_policy_per_ply: list = []                    # [T] of [N] sparse soft policy (idx[],w[]) or None

    alive_mask = torch.ones(num_games, dtype=torch.bool, device=device)
    game_outcome = torch.zeros(num_games, dtype=torch.int32, device=device)
    terminal_threefold = torch.zeros(num_games, dtype=torch.bool, device=device)
    terminal_no_progress = torch.zeros(num_games, dtype=torch.bool, device=device)
    game_length = torch.zeros(num_games, dtype=torch.int32, device=device)
    move_count = torch.zeros(num_games, dtype=torch.int32, device=device)
    # Per-game: did it ever reach a ≤tb_max_pieces (tablebase) position while alive?
    tb_reached = torch.zeros(num_games, dtype=torch.bool, device=device)

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

        # Per-ply stash of in-TB root FENs for the deferred relabel (None unless the
        # deferred branch fills it). Appended once per ply below to keep T-alignment.
        tb_fen_row = None

        # Opening-move computation, shared by the two call sites below.
        # Uniform mode: multinomial over legal moves. Policy mode (ε-mixture):
        # one raw initial_inference (no search), softmax at opening_policy_temp.
        # Targets: ZERO policy rows under the mixture (make_target zero-CE
        # no-op); legacy mode keeps the historical one-hot.
        def _opening_actions():
            uniform_logits = legal_mask.to(torch.float32)
            u_act = torch.multinomial(uniform_logits, 1).squeeze(1).long()
            if mix_on and bool((~open_uniform_t).any()):
                with torch.no_grad():
                    _, pol_logits, _ = network.initial_inference(stacked_obs)
                masked = pol_logits.masked_fill(~legal_mask, float("-inf"))
                probs = torch.softmax(masked / open_policy_temp, dim=-1)
                p_act = torch.multinomial(probs, 1).squeeze(1).long()
                o_act = torch.where(open_uniform_t, u_act, p_act)
            else:
                o_act = u_act
            o_pol = torch.zeros(
                num_games, action_space_size, device=device, dtype=torch.float32)
            if not mix_on:
                o_pol.scatter_(1, o_act.unsqueeze(1), 1.0)  # legacy one-hot
            return o_act, o_pol

        # 3. MCTS — but skip while EVERY game is still inside its opening
        # (per-game open_len; ply < min_open ⇒ all games opening). In the mixed
        # window (min_open <= ply < max_open) the MCTS branch runs and opening
        # games' actions/targets are overridden after it.
        if ply < min_open:
            action, policy = _opening_actions()
            value = torch.zeros(num_games, device=device, dtype=torch.float32)
        else:
            forced_draw_mask = (
                gpu_game.terminal_draw_move_mask(
                    state, legal_mask, min_repeats=term_min_repeats,
                    include_stalemate=term_include_stalemate)
                if use_terminal_draws else None
            )
            # TB relabel targets. Two modes (see defer_relabel above):
            #  - STEERING ON  → run the prober inline; its per-move values bias search.
            #  - STEERING OFF → run only the cheap GPU piece-count gate now and STASH
            #    the in-TB root boards; the value/DTM/policy targets are produced in a
            #    pooled post-game pass. No search bias, so deferral is target-identical.
            steer = None
            if tb_prober is not None and not defer_relabel:
                root_tb_value = tb_prober.root_move_values(
                    state, legal_mask, per_move=need_per_move)
                if tb_prober.last_in_tb is not None:
                    tb_reached |= tb_prober.last_in_tb.to(device) & alive_mask
                steer = root_tb_value if steer_enabled else None
            elif tb_prober is not None:  # deferred
                in_tb = tb_prober.in_tb_mask(state)
                tb_reached |= in_tb.to(device) & alive_mask
                # Reconstruct FENs for the alive, in-TB root boards (cheap vs probing).
                # Appended once per ply in the post-step block (keeps T-alignment).
                tb_fen_row = [None] * num_games
                idxs = torch.nonzero(in_tb & alive_mask, as_tuple=False).flatten().tolist()
                for i in idxs:
                    tb_fen_row[i] = _state_to_board(state, i).fen()
            root_data = mcts.run_batch_gpu(
                stacked_obs, legal_mask, add_noise=True,
                forced_draw_mask=forced_draw_mask,
                root_tb_value=steer,
            )

            if "gumbel_action" in root_data:
                # Plain Gumbel MuZero: A_{n+1} is the deterministic argmax of
                # g + logits + σ(q̂) (exploration lives in the Gumbel draw, so
                # the temperature schedule does not apply) and the policy
                # target is π' = softmax(logits + σ(completedQ)) — the
                # improvement-guaranteed dense target, not visit fractions.
                action = root_data["gumbel_action"]
                policy = root_data["gumbel_policy"]
            else:
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
            # Lc0-faithful: no policy steering by default (tb_steer_policy=False).
            # Self-play is on-policy; the TB signal enters training only via the
            # value relabel (Syzygy WDL) and the moves-left relabel (Gaviota DTM),
            # not by biasing search. Mirrors lc0's rescorer (they rejected the
            # search-side DTZ policy boost). tb_steer_policy=True restores the bias.

        # Mixed opening window (per-game open_len): games still inside their
        # opening take opening actions + zero targets instead of the MCTS
        # results computed above. Merged batches hit this every ply < max_open
        # (seeded games have open_len 0 and need MCTS from ply 0).
        if min_open <= ply < max_open:
            in_open = (open_len_t > ply) & alive_mask                 # [N] bool
            o_action, o_policy = _opening_actions()
            action = torch.where(in_open, o_action, action)
            policy = torch.where(in_open.unsqueeze(1), o_policy, policy)
            value = torch.where(in_open, torch.zeros_like(value), value)

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
        # Per-position TB value target (DTZ-shaped, STM POV; NaN outside TB / during
        # random opening). Deferred path: stash the in-TB root FENs (built pre-step);
        # the targets are produced in the post-game pooled pass. Inline path: read
        # fresh from this ply's root_move_values call.
        if defer_relabel:
            tb_fen_per_ply.append(tb_fen_row if tb_fen_row is not None else [None] * num_games)
        elif (tb_prober is not None and ply >= n_random
                and tb_prober.last_position_value is not None):
            tb_values_per_ply.append(tb_prober.last_position_value.to(torch.float32))
            tb_ml = tb_prober.last_position_moves_left
            tb_moves_left_per_ply.append(
                tb_ml.to(torch.float32) if tb_ml is not None
                else torch.full((num_games,), float("nan"), device=device, dtype=torch.float32))
            # Soft TB policy per game (Python list of (idx[],w[])/None), or all-None if
            # per_move wasn't run this ply (steering+relabel off).
            tb_policy_per_ply.append(
                tb_prober.last_position_policy if tb_prober.last_position_policy is not None
                else [None] * num_games)
        else:
            nanrow = torch.full((num_games,), float("nan"), device=device, dtype=torch.float32)
            tb_values_per_ply.append(nanrow)
            tb_moves_left_per_ply.append(nanrow)
            tb_policy_per_ply.append([None] * num_games)

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

    # 9b. Deferred TB relabel (steering off): one pooled pass over the UNIQUE in-TB
    # FENs gathered during play, then scatter (value, DTM, policy) back into the
    # per-ply accumulators the inline path would have filled. Target-identical; the
    # ~30-probe/position work now overlaps nothing in the GPU hot path and fans out
    # across a CPU process pool instead of serializing one ply at a time.
    if defer_relabel:
        T = len(obs_per_ply)
        params = _prober_params(
            tb_prober, getattr(config, "tb_path", "data/syzygy"),
            getattr(config, "tb_gaviota_path", None))
        all_fens = [f for row in tb_fen_per_ply for f in row if f is not None]
        fenmap = _relabel_fens(
            all_fens, params, relabel_workers, want_policy=tb_policy_relabel)
        n_uniq = len({f for f in all_fens})
        nan = float("nan")
        for t in range(T):
            row = tb_fen_per_ply[t] if t < len(tb_fen_per_ply) else None
            valrow = torch.full((num_games,), nan, dtype=torch.float32)
            mlrow = torch.full((num_games,), nan, dtype=torch.float32)
            polrow = [None] * num_games
            if row is not None:
                for g in range(num_games):
                    fen = row[g]
                    if fen is None:
                        continue
                    res = fenmap.get(fen)
                    if res is None:
                        continue
                    pv, ml, pol = res
                    if pv is not None:
                        valrow[g] = pv
                    if ml is not None:
                        mlrow[g] = ml
                    polrow[g] = pol
            tb_values_per_ply.append(valrow)
            tb_moves_left_per_ply.append(mlrow)
            tb_policy_per_ply.append(polrow)
        tqdm.write(
            f"  TB relabel (deferred): {len(all_fens)} in-TB plies, {n_uniq} unique "
            f"FENs probed across {max(relabel_workers, 1)} worker(s)")

    # 10. Stack and bulk-transfer to CPU. ONE pass.
    obs_stack = torch.stack(obs_per_ply, dim=1)                       # [N, T, C, H, W]
    actions_stack = torch.stack(actions_per_ply, dim=1)               # [N, T]
    policies_stack = torch.stack(policies_per_ply, dim=1)             # [N, T, A]
    values_stack = torch.stack(values_per_ply, dim=1)                 # [N, T]
    rewards_stack = torch.stack(rewards_per_ply, dim=1)               # [N, T]
    legal_masks_stack = torch.stack(legal_masks_per_ply, dim=1)       # [N, T, A]
    tb_values_stack = torch.stack(tb_values_per_ply, dim=1)           # [N, T]
    tb_moves_left_stack = torch.stack(tb_moves_left_per_ply, dim=1)   # [N, T]

    obs_cpu = obs_stack.cpu().numpy()
    actions_cpu = actions_stack.cpu().numpy()
    policies_cpu = policies_stack.cpu().numpy()
    values_cpu = values_stack.cpu().numpy()
    rewards_cpu = rewards_stack.cpu().numpy()
    legal_masks_cpu = legal_masks_stack.cpu().numpy()
    tb_values_cpu = tb_values_stack.cpu().numpy()
    tb_moves_left_cpu = tb_moves_left_stack.cpu().numpy()
    final_obs_cpu = final_obs.cpu().numpy()
    game_length_cpu = game_length.cpu().numpy()
    game_outcome_cpu = game_outcome.cpu().numpy()
    terminal_threefold_cpu = terminal_threefold.cpu().numpy()
    terminal_no_progress_cpu = terminal_no_progress.cpu().numpy()
    tb_reached_cpu = tb_reached.cpu().numpy()
    if tb_prober is not None:
        n_reached = int(tb_reached_cpu.sum())
        # In deferred mode the probing happens in the post-game pooled pass (logged
        # separately), so tb_prober.n_probed stays 0 — report the gate count instead.
        probed_clause = ("relabel deferred (see above)" if defer_relabel
                         else f"{tb_prober.n_probed} root positions probed this batch")
        tqdm.write(
            f"  TB probe: {n_reached}/{num_games} games reached <= {tb_prober.max_pieces} "
            f"pieces; {probed_clause}"
        )
        tb_prober.close()

    # 11. Build GameHistory objects.
    histories: list[GameHistory] = []
    for g in range(num_games):
        L = int(game_length_cpu[g])
        h_g = GameHistory(game_name=config.game)
        if fen_list[g] is not None:
            h_g.start_fen = fen_list[g]   # buffer reconstruction replays from the seed
        h_g.reached_tb = bool(tb_reached_cpu[g])
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
        # Per-ply DTZ-shaped TB value targets (NaN where the ply wasn't in TB).
        # Only attach when the game actually reached the tablebase (saves memory
        # + serialization for the majority that never simplify that far).
        tb_row = tb_values_cpu[g, :L]
        if np.isfinite(tb_row).any():
            h_g.tablebase_values = [float(v) for v in tb_row]
        tbml_row = tb_moves_left_cpu[g, :L]
        if np.isfinite(tbml_row).any():
            h_g.tablebase_moves_left = [float(v) for v in tbml_row]
        # Per-ply soft TB policy (sparse). tb_policy_per_ply is [T] of [N] lists.
        if tb_policy_per_ply:
            pol_row = [tb_policy_per_ply[t][g] for t in range(L)]
            if any(p is not None for p in pol_row):
                h_g.tablebase_policy = pol_row
        h_g.observations.append(torch.from_numpy(final_obs_cpu[g]).clone())
        histories.append(h_g)

    # 12. TB rollout fill (win adjudication by demonstration): won-but-unconverted
    # NON-seeded games are truncated at their first decisive in-TB ply and finished
    # with TB-optimal play. Runs BEFORE resignation (callers apply it after us);
    # filled games are exempted there via h.tb_filled.
    if getattr(config, "tb_rollout_fill", False) and tb_prober is not None:
        _fill_params = _prober_params(
            tb_prober, getattr(config, "tb_path", "data/syzygy"),
            getattr(config, "tb_gaviota_path", None))
        t_fill = time.time()
        n_filled = _apply_tb_rollout_fill(histories, _fill_params)
        if n_filled:
            tqdm.write(
                f"  TB rollout fill: {n_filled}/{num_games} games truncated at their "
                f"first decisive TB ply + finished with TB-optimal play "
                f"({time.time() - t_fill:.1f}s)")

    return histories


_SEED_ARCHIVE_CACHE: dict[str, tuple[list[str], list[int] | None]] = {}


def _load_seed_archive(path: str) -> tuple[list[str], list[int] | None]:
    """Load + cache the endgame-seed FEN archive (one FEN per line) and, when a
    ``<path>.dtm`` sidecar exists (scripts/annotate_seed_dtm.py), the per-seed
    |DTM| annotations (0 = drawn seed, -1 = probe failure)."""
    if path not in _SEED_ARCHIVE_CACHE:
        try:
            with open(path) as f:
                fens = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            fens = []
        dtms = None
        try:
            with open(path + ".dtm") as f:
                vals = [int(ln.strip()) for ln in f if ln.strip()]
            if len(vals) == len(fens):
                dtms = vals
        except Exception:
            dtms = None
        _SEED_ARCHIVE_CACHE[path] = (fens, dtms)
    return _SEED_ARCHIVE_CACHE[path]


def seed_curriculum_dtm_cap(training_step: int, config) -> int | None:
    """Reverse-curriculum DTM cap for seed sampling (Florensa-style: start
    near the goal, expand outward). Ramps linearly from
    ``seed_curriculum_dtm_easy`` to unlimited over
    ``seed_curriculum_anneal_frac`` of training. None = no cap (off, or ramp
    complete)."""
    if not getattr(config, "seed_curriculum", False):
        return None
    frac = float(getattr(config, "seed_curriculum_anneal_frac", 0.5))
    total = max(1, int(getattr(config, "training_steps", 1)))
    p = min(1.0, training_step / max(1.0, frac * total))
    if p >= 1.0:
        return None
    easy = int(getattr(config, "seed_curriculum_dtm_easy", 8))
    hard = int(getattr(config, "seed_curriculum_dtm_hard", 100))
    return int(round(easy + p * (hard - easy)))


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

        # Endgame seeding (Phase 1, resident path only): a fraction of games start
        # from tablebase endgame FENs (on-policy curriculum) — the model plays them
        # out and we relabel with value+DTM truth. Seeded games run as a separate
        # sub-batch (they need n_random=0); normal games run as usual.
        seed_frac = float(getattr(config, "endgame_seed_frac", 0.0))
        seed_arch = getattr(config, "endgame_seed_archive", None)
        seed_fens: list[str] = []
        if use_resident and seed_frac > 0.0 and seed_arch:
            pool, pool_dtms = _load_seed_archive(seed_arch)
            if pool:
                n_seed = int(round(seed_frac * num_games))
                cap = seed_curriculum_dtm_cap(training_step, config)
                if cap is not None and pool_dtms is not None:
                    # Reverse curriculum: draw from seeds solvable within the
                    # current DTM cap (short chains first; the p^N conversion
                    # math is benign there). Drawn seeds (dtm==0) stay eligible
                    # throughout — they feed the draw-hold metric. Fall back to
                    # the full pool if the cap leaves too few candidates.
                    eligible = [f for f, d in zip(pool, pool_dtms)
                                if d == 0 or 0 < d <= cap]
                    if len(eligible) >= max(32, n_seed):
                        pool = eligible
                seed_fens = [random.choice(pool) for _ in range(n_seed)]
        n_normal = num_games - len(seed_fens)

        histories = []
        if (use_resident and seed_fens
                and bool(getattr(config, "merged_seed_batch", False))):
            # MERGED sweep (next-run lever #11): normal + seeded games in the
            # SAME resident batches — one straggler tail and one set of fixed
            # per-ply overheads per chunk instead of two. Interleave so every
            # chunk carries ~the configured seed fraction.
            slots: list = [None] * n_normal + list(seed_fens)
            random.shuffle(slots)
            iterator = range(0, len(slots), n_parallel)
            if show_progress:
                iterator = tqdm(iterator, desc=desc + " (merged)", leave=False)
            for s in iterator:
                chunk = slots[s:s + n_parallel]
                histories.extend(play_fn(
                    network, config, len(chunk), device, training_step,
                    start_fens=chunk))
            return _apply_resignation(histories, config)
        # 1) normal games
        rem = n_normal
        iterator = range(0, n_normal, n_parallel) if n_normal > 0 else []
        if show_progress and n_normal > 0:
            iterator = tqdm(iterator, desc=desc, leave=False)
        for _ in iterator:
            batch = min(n_parallel, rem)
            histories.extend(play_fn(network, config, batch, device, training_step))
            rem -= batch
        # 2) seeded endgame games (start_fens per chunk; resident play_fn only)
        for s in range(0, len(seed_fens), n_parallel):
            chunk = seed_fens[s:s + n_parallel]
            histories.extend(play_fn(
                network, config, len(chunk), device, training_step, start_fens=chunk))
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
