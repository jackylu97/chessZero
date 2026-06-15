"""Hyperparameter configurations per game."""

from dataclasses import dataclass, field


@dataclass
class MuZeroConfig:
    """All hyperparameters for MuZero training."""

    # Game
    game: str = "tictactoe"

    # Network architecture
    hidden_planes: int = 32
    num_residual_blocks: int = 2
    latent_h: int = 3
    latent_w: int = 3
    fc_hidden: int = 64
    value_support_size: int = 10  # half-width of categorical support for value
    reward_support_size: int = 1  # half-width of categorical support for reward
    # Dimension of the per-action learned embedding fed to the dynamics network.
    # Replaces the prior scalar broadcast (action_index / action_space_size, 1 channel),
    # which had only one weight slice in dynamics.conv_in for the action input —
    # architecturally incapable of producing qualitatively different transformations
    # per action. Probed on the 2026_05_07_small_cold step-1000 checkpoint, that
    # collapsed to cos(dynamics(h, a_i), dynamics(h, a_j)) = 0.9999 across 20 distinct
    # chess actions. With ``nn.Embedding(action_space_size, action_embed_dim)``,
    # conv_in gets ``action_embed_dim × hidden_planes × 9`` parameters dedicated to
    # encoding "what does action a do."
    # 16 is the default. Setting ``action_embed_dim = action_space_size`` recovers
    # (un-trained) one-hot — LightZero's default. If 16 under-discriminates after a
    # sanity run, bump toward one-hot. See design.md § Action encoding.
    action_embed_dim: int = 16

    # MCTS
    num_simulations: int = 25
    c_puct: float = 1.25  # unused (dynamic pb_c used instead), kept for reference
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_drop_step: int = 15  # move number to switch to final temp

    # Training
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.25  # downweight value loss (Reanalyze paper).
                                     # Fallback / back-compat: used when the phase-dependent
                                     # fields below are both left at their default (None).
    # Phase-dependent value loss weighting. Stockfish warmstart targets are clean supervised
    # signal (deterministic, teacher >> net, no feedback loop) → high weight. Self-play
    # targets are noisy MCTS bootstraps → paper-standard 0.25 to avoid value-chasing-itself
    # feedback loops. Switch is gated on pool_alive in trainer._train_step. When either
    # value is None, the trainer falls back to the scalar value_loss_weight above so older
    # presets keep working unchanged. See design.md § Deferred: Phase-Dependent value_loss_weight.
    value_loss_weight_warmstart: float | None = None
    value_loss_weight_selfplay: float | None = None
    num_unroll_steps: int = 5  # K: number of future steps to unroll
    td_steps: int = -1  # -1 means use full game return
    discount: float = 1.0  # 1.0 for board games (no discounting)
    training_steps: int = 5000
    checkpoint_interval: int = 500
    log_interval: int = 50

    # Replay buffer
    replay_buffer_size: int = 5000  # number of games
    min_buffer_size: int = 50  # min games before training starts
    # Two-pool FIFO mode. When set, ``replay_buffer_size`` is partitioned into
    # a warmstart pool of this size and a self-play pool of
    # ``replay_buffer_size - warmstart_buffer_size``; each pool evicts FIFO
    # within its own cap. Required for ``warmstart_sample_frac > 0`` to remain
    # effective past pool exhaustion — without it, single-pool FIFO drains
    # warmstart games as self-play arrives and the stratified sampler silently
    # degrades to flat. None = legacy single-pool FIFO.
    warmstart_buffer_size: int | None = None
    # Cap the number of most-recent self-play games persisted to .buf per
    # checkpoint. None = no cap (save everything the in-memory buffer holds).
    # The in-memory buffer is unaffected; only the on-disk snapshot is trimmed.
    max_buf_save_games: int | None = None

    # Self-play
    num_self_play_games: int = 100  # games per self-play batch
    self_play_interval: int = 100  # training steps between self-play rounds
    # Two-phase (Option A) curriculum. For the first ``self_play_warmup_steps``
    # training steps, self-play and reanalyze are gated OFF and the network
    # trains purely on the supervised Stockfish stream (a fixed-length warmstart
    # pretrain). At/after this step they flip on for the rest of training. The
    # warmstart anchor (two-pool ``warmstart_buffer_size``) persists into the
    # self-play phase. 0 = disabled → fall back to the legacy pool-exhaustion
    # gate (self-play turns on when the injection pool runs dry).
    self_play_warmup_steps: int = 0
    num_parallel_games: int = 1    # games to run simultaneously in batched MCTS (1 = serial)
    random_opening_plies: int = 0  # play N random legal moves before MCTS; 0 = disabled

    # AMP
    use_amp: bool = True

    # Evaluation
    eval_games: int = 50
    eval_interval: int = 500

    # Device
    device: str = "cuda"

    # LR scheduling — piecewise constant decay. Empty list = no decay.
    lr_decay_milestones: list = field(default_factory=list)  # fractions of training_steps
    lr_decay_factor: float = 0.1
    # Linear warmup from ~0 → lr over lr_warmup_steps, then hand off to MultiStepLR.
    # 0 = disabled (scheduler is just MultiStepLR). Useful when bumping lr: protects
    # against early-training gradient explosion before amp_scale settles.
    lr_warmup_steps: int = 0

    # Prioritized Experience Replay (PER)
    per_alpha: float = 0.6    # priority exponent: 0 = uniform, 1 = fully prioritized
    per_beta_init: float = 0.4  # IS weight exponent: anneals from this to 1.0 over training
    per_epsilon: float = 1e-6   # min priority floor

    # Temperature scheduling across training steps (within-game scheduling unchanged)
    # List of (step_fraction, temperature) pairs applied to temperature_init
    temperature_schedule: list = field(default_factory=lambda: [(0.5, 0.5), (0.75, 0.25)])

    # Reanalyze — re-run MCTS on stored positions with the current network
    reanalyze_interval: int = 0   # training steps between reanalyze calls; 0 = disabled
    reanalyze_batch_size: int = 20  # number of games to reanalyze per call
    # Use the GPU-resident TensorMCTS path for reanalyze. Default flipped from
    # False → True on 2026-05-08 after observing reanalyze take ~22 min/call
    # (113K positions / numpy BatchedMCTS @ ~12 ms/position) vs the GPU path's
    # estimated ~2 min/call. Tests in tests/test_reanalyze_tensor_mcts.py
    # cover both paths; falls back gracefully (raises NotImplementedError on
    # use_gumbel=True). Independent of ``use_tensor_mcts`` so self-play and
    # reanalyze can be A/B'd separately by flipping either knob.
    # Subtree reuse is N/A for reanalyze (each position is independent).
    reanalyze_use_tensor_mcts: bool = True

    # Stockfish injection — inject pre-generated Stockfish games into the buffer
    # as if they were self-play rounds. Pool (shards of list[GameHistory] with
    # external_values populated) is passed via --stockfish-injection-path.
    # While the pool has games left, self-play and reanalyze are both gated off:
    # the supervised Stockfish stream takes self-play's role; reanalyze on a
    # buffer of external_values-only games is a no-op anyway. When the pool
    # exhausts, self-play and reanalyze flip on automatically for the rest of
    # training. Buffer FIFO + PER handle staleness naturally — identical code
    # path to self-play.
    # Both 0 = disabled.
    stockfish_injection_games: int = 0       # games per injection round
    stockfish_injection_interval: int = 0    # training steps between rounds

    # Categorical value/reward target encoding uses h(x) = sign(x)(sqrt(|x|+1)-1)
    # + 0.001x to compress heavy-tailed Atari-style returns. For bounded chess
    # targets in [-1,+1], h(x) compresses them into ~3 of 21 bins (roughly 0.415,
    # 0, -0.415), throwing away the resolution the bins were meant to provide.
    # Set False to skip the transform (identity) — targets then span the full
    # support at ~0.1 resolution. Both muzero-general and LightZero apply h(x)
    # unconditionally; flip this for board games where targets are already bounded.
    use_scalar_transform: bool = True

    # Linear scale applied to raw value targets before bin encoding (and inverse
    # applied after decoding). Only active when use_scalar_transform=False — h(x)
    # and linear scaling are mutually exclusive encodings. For bounded targets in
    # [-R, +R] paired with value_support_size=S, setting value_target_scale=S/R
    # spreads raw values across the full support so every bin carries gradient.
    # Chess example: raw targets in [-1,+1], support_size=2 → scale=2 → 5 bins
    # at {-1, -0.5, 0, +0.5, +1} in raw-value terms.
    value_target_scale: float = 1.0

    # Value head type — 'support' (default, MuZero paper) or 'wdl' (Lc0).
    #   support: 2*value_support_size+1 bins over a discrete support; trained
    #            via cross-entropy with two-hot scalar-target encoding.
    #   wdl:     3-output (Win, Draw, Loss) classifier; trained on one-hot
    #            game outcome from the side-to-move's POV. Decoded to scalar
    #            via V = P(W) - P(L) + draw_score · P(D). Best fit for chess
    #            (canonical class structure: {-1, 0, +1} game outcomes).
    value_head_type: str = "support"

    # Draw shaping for the WDL head's scalar conversion (search-time only):
    #   V = P(W) - P(L) + draw_score · P(D)
    # Lc0 default 0.0. Negative values penalize draws (anti-draw shaping);
    # paper-faithful is 0.0. Does NOT affect training targets — only the
    # scalar exposed to MCTS PUCT and Q backups.
    draw_score: float = 0.0

    # q-blend ratio for WDL training targets (applied in BOTH phases by
    # GameHistory.make_target / _wdl_target_at). q_ratio (q) is the weight on
    # the "blended-in" signal; (1-q) is the weight on the phase's legacy target.
    # Blending two probability distributions with weights summing to 1 yields a
    # valid distribution.
    #   WARMSTART ply (external_values present):
    #     target = (1-q)·eval_to_wdl(external_value) + q·outcome_onehot
    #     i.e. blend the GAME OUTCOME into the Stockfish-eval target. This
    #     de-saturates warmstart value targets so the head reads won positions
    #     decisively rather than washing them toward the eval's draw zone.
    #   SELF-PLAY ply (no external_values, or ply past their end):
    #     target = (1-q)·outcome_onehot + q·eval_to_wdl(root_value)
    #     i.e. blend the MCTS ROOT VALUE (scalar root_values[i], STM POV, mapped
    #     to WDL via eval_to_wdl) into the outcome one-hot — the Lc0 q-blend.
    #     Falls back to pure outcome_onehot if root_values is missing/short.
    # At q_ratio == 0.0 this reduces EXACTLY to the legacy behavior (warmstart →
    # pure eval_to_wdl; self-play → pure outcome one-hot) — back-compat guarantee.
    # Lc0 default 0.0. Higher values mitigate the credit-assignment problem on
    # noisy self-play outcomes.
    q_ratio: float = 0.0

    # Split q_ratio per phase (override the single q_ratio above when set).
    # The two phases blend in DIFFERENT signals and carry different risk:
    #   warmstart_q_ratio — blends the GAME OUTCOME into an EXTERNAL Stockfish
    #     eval target. Not self-referential, so it can run hot (≈0.5) to
    #     de-saturate the eval's draw zone with decisive results.
    #   selfplay_q_ratio  — blends the network's OWN MCTS root value into the
    #     outcome (the literal Lc0 q-blend). Self-referential → a feedback-loop
    #     risk, so keep it cool (≈0.1; AlphaZero/Lc0 default 0.0).
    # When None, the phase falls back to the single ``q_ratio`` above.
    warmstart_q_ratio: float | None = None
    selfplay_q_ratio: float | None = None

    # eval_to_wdl conversion (warmstart-only, value_head_type="wdl"): convert
    # Stockfish per-position eval (in [-1,+1] STM POV) into a soft (P_W, P_D,
    # P_L) target via parameterized 3-way logistic. ``alpha`` controls
    # sharpness (4.0 ≈ Stockfish-cp-like steepness); ``beta`` controls draw-
    # zone width (2.0 ≈ P_D=0.76 at eval=0). Restores per-position teacher
    # signal density during warmstart that pure-z one-hot targets discarded.
    eval_to_wdl_alpha: float = 4.0
    eval_to_wdl_beta: float = 2.0

    # Stratified sampling: at every training batch, sample
    # floor(batch_size * warmstart_sample_frac) games from warmstart-only
    # (games with external_values populated) and the rest from self-play.
    # Maintains a permanent Stockfish-anchor in training even after the
    # buffer's FIFO eviction has flipped the in-memory composition entirely
    # to self-play. Directly attacks the catastrophic-forgetting / drawish-
    # basin failure observed in runs 0001 / 0002 (post-pool-exhaustion drift).
    # 0.0 = back-compat (no stratification). 0.3-0.5 = chess recommended.
    warmstart_sample_frac: float = 0.0

    # Decisive-game resampling: at every batch, sample floor(batch_size *
    # decisive_sample_frac) games from DECISIVE self-play games (|game_outcome|=1,
    # i.e. |z|=1) and the rest from the others (draws). Keeps a non-constant value
    # signal in every batch when the self-play draw rate climbs and the z=0 majority
    # would otherwise wash out the value gradient (the draw-saturation loop:
    # flat value -> safe-draw play -> all-draw targets -> flat value). The cheap,
    # self-play-only seed for the value head (vs Stockfish warmstart's external
    # seed). Mutually exclusive with warmstart_sample_frac (warmstart wins). Falls
    # back to flat PER when 0.0 or no decisive games exist yet. 0.5 = aggressive
    # (half the batch decisive vs a natural ~8-15%); tune down toward 0.3 if it
    # overfits the small decisive-game set. See deferred_decisive_game_prioritization.
    decisive_sample_frac: float = 0.0

    # AlphaZero-style history encoding: stack the last N ply observations
    # along the channel dimension before passing to the network. Newest
    # frame first. Missing frames (early game) zero-padded.
    #   1: current state only (no history; pre-history-encoding behavior).
    #   8: AlphaZero/Lc0 default for chess. Lets the network perceive
    #      threefold repetition and 50-move-rule progress, which a stateless
    #      encoder cannot. Reconstructed at sample time from the stored
    #      per-ply observations — no disk impact on existing data.
    history_frames: int = 1

    # Root-heavy loss weighting (MuZero paper / muzero-general pseudocode):
    # root prediction gets weight 1.0, each of the K unroll steps gets weight 1/K.
    # Default False uses the current uniform 1/(K+1) weighting (≈ LightZero's
    # convention). Flip to True for an A/B against the paper-prescribed shape;
    # effectively increases root-level gradient ~6× vs tail (K=5) and ~doubles total
    # loss magnitude, so watch train/grad_norm and train/amp_scale on rollout.
    use_root_heavy_loss: bool = False

    # Sampled MuZero (Hubert 2021, Proposed Modification): sample K distinct
    # actions per node via Gumbel-Top-K; PUCT prior is π_net renormalized over σ,
    # training target is raw N(a)/ΣN(a). Required for large action spaces (chess).
    # None = expand all legal actions at root / all actions at leaves (tiny action spaces only).
    sample_k: int | None = None

    # Gumbel MuZero (Danihelka 2022, Plain variant): Sequential Halving + Gumbel-Top-K
    # at root; PUCT untouched at non-root. Training target is π' = softmax(logits +
    # σ(completedQ)), loss is KL(π', π_net). No Dirichlet noise (Gumbel provides
    # exploration). See plan_gumbel_muzero.md and src/mcts/gumbel.py.
    use_gumbel: bool = False
    gumbel_num_considered: int = 16      # m: actions sampled without replacement at root
    gumbel_c_visit: float = 50.0         # σ transform (paper Eq. 8)
    gumbel_c_scale: float = 1.0
    use_gumbel_noise: bool = True        # True for training self-play; False for eval

    # EfficientZero consistency loss (Ye 2021). SimSiam-style self-supervision on the
    # dynamics net: rolled-out latent must match representation(future_obs). Supervises
    # the world model directly, decoupled from noisy value/reward signal — helps most
    # during cold start when value targets are starved. See src/model/muzero_net.py
    # (ProjectionNetwork / PredictionHead) and lzero/policy/efficientzero.py.
    use_consistency_loss: bool = False
    consistency_loss_weight: float = 2.0  # LightZero ssl_loss_weight default
    proj_hid: int = 1024
    proj_out: int = 1024
    pred_hid: int = 512
    pred_out: int = 1024
    # Consistency target: use a single-frame (newest-ply-only, zero-padded to the
    # full history-channel count) observation for the SimSiam target instead of the
    # full T-frame stack. With history_frames=8 the full-stack target shares 7/8
    # frames between adjacent unroll steps (~88% trivial), letting an identity
    # dynamics satisfy consistency. NOTE (2026-06-02): de-risking probes show this
    # alone does NOT fix action-blindness (the dynamics still memorizes h→next and
    # ignores the action); it only de-trivializes the target / grounds the world
    # model. The action-awareness fix is use_inverse_dynamics_loss below. See
    # dynamics_gradient_starvation memory note.
    consistency_single_frame_target: bool = False

    # Inverse-dynamics auxiliary loss (ICM/Pathak 2017). A small head predicts the
    # action a_k from (h_k, h_{k+1}=dynamics(h_k,a_k)); CE against the true action.
    # The action is only recoverable if the dynamics output depends on it, so this
    # is a non-bypassable pressure forcing action-conditioned dynamics — the
    # validated fix for the action-blind collapse (cos(dyn_a,dyn_b)≈1.0). Probe:
    # drives cross-action cos 1.0→0.61 where every consistency variant stayed ~1.0.
    # Training-only head; unused by MCTS. See scripts/probe_fix_candidates.py.
    use_inverse_dynamics_loss: bool = False
    inverse_dynamics_loss_weight: float = 1.0
    inverse_dynamics_hidden: int = 256

    # Value-head output-layer init std. 0.0 = zero-init (MuZero/LightZero default;
    # blocks gradient to the body at cold start because Wᵀ·grad_out = 0). A small
    # positive std (e.g. 0.01) lets value-head gradient reach the representation/
    # dynamics body from step 0 without destabilizing early MCTS. Mitigation for the
    # cold-start gradient block; not a substitute for a body-direct signal.
    value_head_init_std: float = 0.0

    # GPU-resident chess env. When True (chess only), self-play env ops
    # (to_tensor / legal_actions / step) run as batched torch ops via
    # GpuChessGame instead of per-game python-chess calls. The MCTS itself
    # is unchanged. Cross-validated against python-chess on 50k random
    # positions and 1k random-play games. See plan_gpu_chess_engine.md.
    use_gpu_chess: bool = False

    # Tensor-native MCTS. When True, the parallel batched self-play paths
    # use TensorMCTS (GPU-resident tree, 0 syncs/sim, 1 sync/ply) instead of
    # BatchedMCTS (numpy tree, 3 syncs/sim). Currently supports the Sampled
    # MuZero §5.1 Proposed Modification only; Gumbel root is not yet
    # implemented in the tensor path. See plan_tensor_mcts_implementation.md.
    use_tensor_mcts: bool = False
    # Fully GPU-resident self-play loop. Requires use_tensor_mcts + use_gpu_chess.
    # Eliminates per-ply CPU↔GPU syncs at the env/MCTS boundary (currently ~12
    # per ply); the entire batch's history lives on the GPU until ONE bulk
    # transfer at end-of-batch. Available for chess only (only game with a GPU
    # batched env). Implementation: src/training/self_play.py::play_games_parallel_gpu_resident.
    use_gpu_resident_self_play: bool = False
    # Storage dtype for the per-node hidden states in TensorMCTS. The
    # node_hidden tensor is by far the dominant memory ([N, M, C, H, W]); at
    # chess preset (N=256, M=401, C=256, H=W=8) it's ~6.7 GB at fp32 and
    # ~3.4 GB at fp16. fp32 is safest; fp16 trades a small precision loss
    # on stored latents (cast back to compute dtype before each
    # recurrent_inference call) for ~2× memory headroom. "float32" or
    # "float16".
    tensor_mcts_hidden_dtype: str = "float32"
    # PUCT selection backend. "compile" (default): torch.compile + inductor
    # fusion. "triton": custom fused per-walk kernel; one launch per sim
    # runs the entire depth walk on-device (1.20× faster than compile on
    # 4090 / chess preset). "eager": plain PyTorch, for debugging.
    tensor_mcts_select_backend: str = "compile"
    # MCTS subtree reuse across plies. When True, after each move the chosen
    # child's subtree is compacted to slot 0 and carried into the next ply's
    # search — preserving its visit counts + value estimates so PUCT starts
    # from an already-developed tree. Doubles the per-game tree storage
    # (M = 2*num_simulations+1) to fit the carry-over plus a fresh ply.
    # Falls back to a fresh search whenever the chosen action wasn't
    # materialized or the subtree+new-sims would overflow M.
    tensor_mcts_subtree_reuse: bool = False
    # Inference autocast dtype for TensorMCTS network forward calls. None
    # disables; "float16" enables fp16 autocast on cuda (Ampere/Ada tensor
    # cores). At production net (256×16) this is the single largest network-
    # forward speedup available — ~1.3-1.5× vs fp32. Tree storage is unaffected
    # (controlled separately by tensor_mcts_hidden_dtype).
    tensor_mcts_amp_dtype: str | None = None
    # Whether to save the replay buffer (.buf) alongside the network
    # checkpoint (.pt) at each save interval. When False, only .pt is
    # saved; resume always cold-starts the buffer via self-play. Useful
    # when buffer save/load is unstable (e.g., compact-format encoding bug
    # observed 2026-05-07 — illegal moves in stored actions due to a
    # chess_gpu pin-detection mismatch). Costs ~30 min cold-start self-play
    # per resume in exchange for skipping buffer corruption.
    save_buffer: bool = True
    # torch.compile the network's inference methods (initial_inference_logits,
    # recurrent_inference_logits, initial_inference, recurrent_inference).
    # Uses mode='default' — pure inductor fusion, no CUDA graphs (cudagraphs
    # mode crashed with TensorMCTS internal compile in earlier benches). Wins
    # come from kernel fusion across conv/norm/relu boundaries; ~1.3× speedup
    # at production net. Compile happens after the network is on-device but
    # the underlying nn.Module is unchanged, so checkpoints save/load cleanly.
    compile_network: bool = False

    # In-loop representation-informativeness probe (repr/* TensorBoard scalars).
    # Every ``repr_probe_interval`` steps, freeze the net and measure how linearly
    # decodable Stockfish eval / game outcome is from the representation on a fixed
    # held-out probe set (plus cross-position cosine + effective rank as collapse
    # guards). r2_eval is the leading indicator of draw-basin collapse — it falls
    # while the value LOSS can still look healthy (see bug_hunt_2026_06_13.md and
    # scripts/probe_representation.py). 0 = disabled. Costs a few seconds per call,
    # so use a long interval. ``repr_probe_positions`` = held-out set size.
    repr_probe_interval: int = 0
    repr_probe_positions: int = 768

    # Multi-game (Phase 2)
    multi_game: bool = False
    games: list[str] = field(default_factory=lambda: ["tictactoe"])
    game_id_embedding_dim: int = 16


def get_config(game: str) -> MuZeroConfig:
    """Return game-specific config with tuned hyperparameters."""
    configs = {
        "tictactoe": MuZeroConfig(
            game="tictactoe",
            hidden_planes=32,
            num_residual_blocks=2,
            latent_h=3, latent_w=3,
            fc_hidden=64,
            num_simulations=100,
            batch_size=64,
            training_steps=5000,
            replay_buffer_size=5000,
            min_buffer_size=50,
            num_self_play_games=100,
            self_play_interval=100,
            lr=1e-3,
            dirichlet_alpha=0.3,
            num_parallel_games=64,
            reanalyze_interval=100,
            reanalyze_batch_size=100,
        ),
        "connect4": MuZeroConfig(
            game="connect4",
            hidden_planes=64,
            num_residual_blocks=4,
            latent_h=6, latent_w=7,
            fc_hidden=128,
            num_simulations=50,
            batch_size=256,
            training_steps=100000,
            replay_buffer_size=50000,
            min_buffer_size=200,
            num_self_play_games=200,
            self_play_interval=200,
            lr=1e-3,
            dirichlet_alpha=0.3,
        ),
        "chess": MuZeroConfig(
            game="chess",
            # Iteration-sized net (2026-05-07). Halved from the production preset
            # (256×16, fc=256) to ~128×8, fc=128 for ~4× faster network forward
            # (half channels × half depth). Trades final strength for turnaround
            # time during config/loss/data-pipeline iteration. Bump back up once
            # the recipe is dialed in.
            hidden_planes=128,
            num_residual_blocks=8,
            latent_h=8, latent_w=8,
            fc_hidden=128,
            num_simulations=200,     # 200 paired with the smaller net for fast iteration.
                                     # Production setpoint was 400 (AlphaZero used 800).
                                     # Halve self-play wall-time vs the production preset.
            batch_size=256,
            training_steps=150000,
            checkpoint_interval=1000,
            lr_decay_milestones=[0.5, 0.75],  # decay 10× at 50k and 75k
            lr_warmup_steps=500,               # ramp up to lr over first 500 steps;
                                               # AMP scale is still stabilizing there.
            replay_buffer_size=1500,    # 2026-06-12: restored to 1500 (was halved to 750 on
                                        # 2026-05-08 to fit the 24 GB WSL RSS ceiling). The
                                        # run box now has 251 GB RAM; at ~700-ply avg ×
                                        # ~50 KB/ply, 1500 games ≈ 52 GB peak — comfortable.
                                        # 2× the buffer also halves sample staleness pressure
                                        # at the same self-play cadence.
            warmstart_buffer_size=None, # Cold-start mode (2026-05-07): no warmstart pool,
                                        # all 2500 slots available for self-play games.
                                        # Pair with stockfish_injection_games=0 and
                                        # warmstart_sample_frac=0.0 below.
            min_buffer_size=500,
            num_self_play_games=256,   # one num_parallel_games sweep per round
            self_play_interval=512,   # 2:1 train:selfplay ratio
            lr=1e-3,                   # 2026-06-12: dropped from 2e-3 back to the paper
                                        # floor for this run. With 500-step warmup (below).
                                        # See design.md § Deferred: Double Base Learning
                                        # Rate for Chess for the 2e-3 reasoning if re-raising.
            value_loss_weight_warmstart=1.0,  # clean Stockfish targets: strong supervision
            value_loss_weight_selfplay=1.0,   # MC returns (td_steps=-1) → no bootstrap noise; WDL
                                               # cross-entropy is naturally small (max ln(3)≈1.1).
                                               # Bumped from 0.25: AlphaZero/Lc0 reference setpoint;
                                               # value head was stuck in constant-prediction local
                                               # min at 0.25 (pred_std/target_std ≈ 0.03 at step 1k).
            value_head_type="wdl",      # Lc0-style 3-output W/D/L classifier; replaces
                                        # the 5-bin scalar head. Targets game outcome z directly,
                                        # eliminating the predict-zero collapse failure mode where
                                        # the scalar head settled on the central bin.
            draw_score=-0.05,           # Anti-draw MCTS shaping (Lc0 DrawScore knob,
                                        # tree-side only). Q = WL + draw_score · D, so drawish
                                        # positions get a small negative push during search,
                                        # encouraging decisive moves over solid draws. Doesn't
                                        # affect training targets.
            q_ratio=0.5,                # Fallback when the split knobs below are None.
            warmstart_q_ratio=0.5,      # HOT: blend 50% game outcome into the Stockfish-eval
                                        # WDL — de-saturates the value head on won positions.
                                        # Safe to run hot: the teacher signal is EXTERNAL, not
                                        # self-referential.
            selfplay_q_ratio=0.1,       # COOL: only 10% MCTS-root-value blended into the
                                        # outcome one-hot. The root value is the network's OWN
                                        # estimate (self-referential → feedback-loop risk), so
                                        # keep near the AlphaZero/Lc0 pure-outcome default (0.0).
            warmstart_sample_frac=0.0,  # Cold-start mode (2026-05-07): no warmstart anchor.
                                        # Bump back to 0.4 when re-enabling the Stockfish pool.
            decisive_sample_frac=0.5,   # 2026-06-02: decisive-game resampling seed for the value
                                        # head. The run hit the draw-saturation loop (self-play draw
                                        # rate 0.76->0.92 by step 2048, value head flat). Force half
                                        # the batch from |z|=1 games so the value head keeps a
                                        # non-constant target to learn from. Cheap self-play-only seed;
                                        # q_ratio (the amplifier) and Stockfish warmstart are the
                                        # follow-on levers if this isn't enough.
            history_frames=8,           # AlphaZero canonical: 8 ply frames stacked. Lets the
                                        # network perceive threefold repetition + 50-move-rule
                                        # progress (a stateless encoder cannot). Reconstructed
                                        # at sample time from per-ply obs — no disk inflation.
            dirichlet_alpha=0.1,        # Compromise between 0.03 (Go's value, was over-sharp for
                                        # chess) and 0.3 (AlphaZero chess default, over-disperses
                                        # our soft-MultiPV diffuse prior — produced 88% draws in
                                        # run 2026_04_25_0001 vs 76% with 0.03). α=0.1 keeps real
                                        # exploration noise without compounding policy diffusion.
                                        # Tune up toward 0.3 once WDL + the visit-count loop
                                        # sharpen the policy.
            td_steps=-1,                # Monte Carlo (full game return). Required by WDL —
                                        # n-step bootstrap doesn't compose with categorical
                                        # outcome targets. Matches AlphaZero / Lc0 design.
            random_opening_plies=8,   # play N random legal moves before MCTS kicks in;
                                        # diversifies self-play openings (untrained net
                                        # produces near-identical games from start pos).
                                        # 8 plies ≈ 4 moves/side, reaching common structures.
            temperature_drop_step=30,
            reanalyze_interval=1024,   # keep 1:1 with self_play_interval
            reanalyze_batch_size=256,
            num_parallel_games=256,    # matches training batch_size; batched-sync run_batch
            sample_k=50,               # Sampled MuZero: sample K distinct actions per node (Hubert 2021 Proposed Modification)
            eval_interval=5000,
            repr_probe_interval=2000,  # log repr/* informativeness metrics every 2k steps
            use_consistency_loss=True, # EfficientZero SimSiam consistency loss on dynamics rollouts
            consistency_single_frame_target=True,  # corrected default: 8-frame stack target was
                                                    # ~88% action-invariant (collapse-prone). Grounds
                                                    # the world model; NOT the action-awareness fix.
            use_inverse_dynamics_loss=True,  # ICM inverse model — the validated fix for action-blind
                                             # dynamics (probe: cross-action cos 1.0→0.61). Predicts
                                             # a_k from (h_k, h_{k+1}); forces action-conditioned dynamics.
            inverse_dynamics_loss_weight=1.0,
            value_head_init_std=0.0,    # #4 cold-start gradient-block mitigation; off by default,
                                        # enable via --value-head-init-std 0.01 to A/B.
            stockfish_injection_games=0,       # Cold-start mode (2026-05-07): no Stockfish injection.
            stockfish_injection_interval=0,    # Self-play and reanalyze run from step 0.
            per_alpha=0.6,               # 2026-05-07: re-enabling PER for cold-start.
                                        # Prior reasoning ("DeepMind uses uniform sampling, TD errors
                                        # cluster near zero in draw basin") was backwards — TD errors
                                        # cluster near zero ON DRAWN samples, but decisive samples
                                        # the network mis-predicts have TD ≈ 1.0. PER oversamples
                                        # exactly those, providing the data rebalance we'd otherwise
                                        # need warmstart for. Standard PER paper default α=0.6.
            use_scalar_transform=False,  # chess values live in [-1,+1]; h(x) would collapse them onto bin 0
            value_support_size=2,        # 5 bins at {-2,-1,0,+1,+2}; paired with value_target_scale=2.0 gives {-1,-0.5,0,+0.5,+1} in raw space
            value_target_scale=2.0,      # spread raw [-1,+1] targets across the full 5-bin support
            use_gumbel=False,            # Classic PUCT + Dirichlet for chess. m=16 sampled-root truncation in
                                         # Gumbel MuZero structurally hides tactical moves ranking 17+ in an
                                         # undertrained policy (avg 30-40 legal chess moves in middlegame), and
                                         # Gumbel's advantages were validated on Go where priors are sharper.
                                         # Code path stays flag-gated. See design.md § Deferred: Drop Gumbel.
            # Self-play fast path (default ON for chess). 18× over the legacy
            # BatchedMCTS+python-chess path on a 4090 at our preset.
            #   use_gpu_chess: GpuChessGame batched env (replaces python-chess).
            #   use_tensor_mcts: GPU tensor MCTS (TensorMCTS) instead of BatchedMCTS.
            #   use_gpu_resident_self_play: 0-sync self-play loop, end-of-batch transfer.
            #   tensor_mcts_select_backend='triton': fused PUCT-walk Triton kernel.
            #   tensor_mcts_subtree_reuse: carry chosen child's subtree across plies
            #     (search-quality boost; M doubled to fit carry-over + new sims).
            #   tensor_mcts_hidden_dtype='float16': halves node_hidden memory.
            use_gpu_chess=True,
            use_tensor_mcts=True,
            use_gpu_resident_self_play=True,
            tensor_mcts_select_backend="triton",
            #   tensor_mcts_subtree_reuse=False: disabled 2026-05-07 after
            #     buffer-corruption investigation. _refresh_reused_root's
            #     legal-mask filter on carried-over slots passed code review
            #     but empirically illegal actions (pinned-bishop moves)
            #     ended up with high visit weight in MCTS root at ply 185 of
            #     game 3 in checkpoint_3000.buf. Defensive disable until the
            #     mismatch between code-review and observed behavior is
            #     understood. Costs perf-neutral ~0% (the carried-over speedup
            #     was already marginal — see plan_tensor_mcts.md).
            tensor_mcts_subtree_reuse=False,
            tensor_mcts_hidden_dtype="float16",
            #   tensor_mcts_amp_dtype="float16": fp16 autocast on the network
            #     forward (initial + recurrent inference). At production net
            #     (256 planes × 16 blocks) this dominates per-ply time;
            #     ~1.3-1.5× speedup over fp32 on Ampere/Ada tensor cores.
            #   compile_network=False: torch.compile disabled. Originally turned
            #     on for ~1.3-1.7× on inference path, but observed reproducible
            #     SIGILL crash on 2026-05-07 during reanalyze — BatchedMCTS calls
            #     into the compiled bound methods from a different context (no_grad
            #     decorator + different invocation pattern than self-play), forcing
            #     dynamo recompilation and tripping a fault. Disabled until we
            #     have a clean compile-and-skip-on-recompile gate. AMP autocast
            #     still gives ~1.3-1.5×.
            tensor_mcts_amp_dtype="float16",
            compile_network=False,
            #   save_buffer=True: re-enabled 2026-06-12. Was disabled 2026-05-07
            #     when the v3 compact format hit "illegal move on load" errors
            #     traced to a chess_gpu pin-detection edge case (bishop pinned
            #     to king along rank); that engine bug has since been fixed
            #     (see run 2026_05_08_per_fix "post-chess_gpu-bug-fix"). If
            #     load errors reappear on resume, flip back to False — the
            #     loader tolerates a missing/corrupt .buf by cold-starting.
            save_buffer=True,
        ),
        "checkers": MuZeroConfig(
            game="checkers",
            hidden_planes=128,
            num_residual_blocks=8,
            latent_h=8, latent_w=8,
            fc_hidden=128,
            num_simulations=100,
            batch_size=256,
            training_steps=200000,
            replay_buffer_size=50000,
            min_buffer_size=200,
            num_self_play_games=200,
            self_play_interval=300,
            lr=5e-4,
            dirichlet_alpha=0.1,
        ),
    }
    return configs.get(game, configs["tictactoe"])
