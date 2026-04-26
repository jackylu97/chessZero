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
    # Cap the number of most-recent self-play games persisted to .buf per
    # checkpoint. None = no cap (save everything the in-memory buffer holds).
    # The in-memory buffer is unaffected; only the on-disk snapshot is trimmed.
    max_buf_save_games: int | None = None

    # Self-play
    num_self_play_games: int = 100  # games per self-play batch
    self_play_interval: int = 100  # training steps between self-play rounds
    num_parallel_games: int = 1    # games to run simultaneously in batched MCTS (1 = serial)

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

    # MCTS Q-blend ratio for WDL training targets:
    #   target = q_ratio · q_mcts + (1 - q_ratio) · z
    # where z is the one-hot game outcome and q_mcts is the MCTS root WDL
    # captured during self-play. Lc0 default 0.0 (pure z, Monte Carlo).
    # Higher values mitigate the credit-assignment problem on noisy self-play
    # outcomes. Requires storing MCTS root WDL in GameHistory (currently we
    # store only scalar root_values, so q_ratio>0 needs a schema extension).
    q_ratio: float = 0.0

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
            hidden_planes=256,
            num_residual_blocks=16,
            latent_h=8, latent_w=8,
            fc_hidden=256,
            num_simulations=400,     # 400 to give Q-backups room to dominate the diffuse prior
                                     # at our policy strength. Doubled from 200 for the WDL+history
                                     # bundle (2026-04-25); production AlphaZero used 800. Trade:
                                     # ~2× self-play wall-time per batch.
            batch_size=256,
            training_steps=150000,
            checkpoint_interval=1000,
            lr_decay_milestones=[0.5, 0.75],  # decay 10× at 50k and 75k
            lr_warmup_steps=500,               # ramp up to lr over first 500 steps;
                                               # AMP scale is still stabilizing there.
            replay_buffer_size=2500,    # memory-capped, not game-capped: self-play games
                                        # avg 277 plies × ~25 KB/ply ≈ 6.7 MB/game, so
                                        # 2500 slots peaks at ~17 GB RAM — safe on a
                                        # 32 GB host (was 5000 → ~34 GB peak, OOM'd
                                        # around step 36k of 2026_04_22_0002 as self-play
                                        # games displaced shorter stockfish ones).
                                        # Revert to 5000 once compact GameHistory
                                        # encoding lands (see plan_compact_gamehistory_encoding.md).
            min_buffer_size=500,
            num_self_play_games=256,   # one num_parallel_games sweep per round
            self_play_interval=512,   # 2:1 train:selfplay ratio
            lr=2e-3,                   # 2× paper floor; grad_norm + amp_scale confirm
                                        # optimizer has headroom at 1e-3. With 500-step
                                        # warmup (below). See design.md § Deferred: Double
                                        # Base Learning Rate for Chess.
            value_loss_weight_warmstart=1.0,  # clean Stockfish targets: strong supervision
            value_loss_weight_selfplay=0.25,  # noisy MCTS bootstraps: paper-standard damp
            value_head_type="wdl",      # Lc0-style 3-output W/D/L classifier; replaces
                                        # the 5-bin scalar head. Targets game outcome z directly,
                                        # eliminating the predict-zero collapse failure mode where
                                        # the scalar head settled on the central bin.
            draw_score=-0.05,           # Anti-draw MCTS shaping (Lc0 DrawScore knob,
                                        # tree-side only). Q = WL + draw_score · D, so drawish
                                        # positions get a small negative push during search,
                                        # encouraging decisive moves over solid draws. Doesn't
                                        # affect training targets.
            q_ratio=0.0,                # Lc0 default; pure z target.
            warmstart_sample_frac=0.4,  # 40% of every training batch comes from warmstart games
                                        # (external_values populated). Permanent Stockfish anchor —
                                        # prevents catastrophic forgetting during self-play handoff
                                        # that drove the basin collapse in runs 0001/0002.
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
            temperature_drop_step=30,
            reanalyze_interval=1024,   # keep 1:1 with self_play_interval
            reanalyze_batch_size=256,
            num_parallel_games=256,    # matches training batch_size; batched-sync run_batch
            sample_k=50,               # Sampled MuZero: sample K distinct actions per node (Hubert 2021 Proposed Modification)
            eval_interval=5000,
            use_consistency_loss=True, # EfficientZero SimSiam consistency loss on dynamics rollouts
            stockfish_injection_games=256,     # 256 games every 240 steps ≈ 1.07 g/step
            stockfish_injection_interval=240,  # ~3 center-touches per position (×6 unroll = ~18 loss terms);
                                               # with a 32k-game pool this runs ~30k steps before exhaustion,
                                               # at which point self-play + reanalyze flip on automatically
            use_scalar_transform=False,  # chess values live in [-1,+1]; h(x) would collapse them onto bin 0
            value_support_size=2,        # 5 bins at {-2,-1,0,+1,+2}; paired with value_target_scale=2.0 gives {-1,-0.5,0,+0.5,+1} in raw space
            value_target_scale=2.0,      # spread raw [-1,+1] targets across the full 5-bin support
            use_gumbel=False,            # Classic PUCT + Dirichlet for chess. m=16 sampled-root truncation in
                                         # Gumbel MuZero structurally hides tactical moves ranking 17+ in an
                                         # undertrained policy (avg 30-40 legal chess moves in middlegame), and
                                         # Gumbel's advantages were validated on Go where priors are sharper.
                                         # Code path stays flag-gated. See design.md § Deferred: Drop Gumbel.
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
