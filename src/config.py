"""Hyperparameter configurations per game."""

from dataclasses import dataclass, field, replace


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
    # Hard ply cap for a self-play game (draw if reached). Also bounds the GPU-resident
    # self-play per-ply accumulators (obs/policy/legal-mask stacks grow with the LONGEST
    # game in the batch) — the dominant memory cost, so lowering it cuts both wall-clock
    # and peak VRAM. Mirrors GpuChessGame.max_plies (set on the env at construction).
    max_plies: int = 1024
    # Two-phase (Option A) curriculum. For the first ``self_play_warmup_steps``
    # training steps, self-play and reanalyze are gated OFF and the network
    # trains purely on the supervised Stockfish stream (a fixed-length warmstart
    # pretrain). At/after this step they flip on for the rest of training. The
    # warmstart anchor (two-pool ``warmstart_buffer_size``) persists into the
    # self-play phase. 0 = disabled → fall back to the legacy pool-exhaustion
    # gate (self-play turns on when the injection pool runs dry).
    self_play_warmup_steps: int = 0
    # Optional FRACTIONAL form of the above, for notation consistency with the
    # batch-mixture schedule (which is expressed in fractions of training_steps).
    # When set (not None), resolves to self_play_warmup_steps = round(frac *
    # training_steps) at launch, so the self-play-generation boundary and the
    # mixture's self-play-onset scale together instead of drifting apart when
    # training_steps changes (the 30k-vs-60k misalignment). None = use the
    # absolute self_play_warmup_steps above.
    self_play_warmup_frac: float | None = None
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
    # cover both paths. Under use_gumbel=True the tensor path routes
    # run_batch_gpu (the only tensor entry with the Gumbel root); its
    # gumbel_policy/root_value match the numpy select_action_gumbel write path
    # (_gumbel_finalize is the batched port of select_action_gumbel).
    # Independent of ``use_tensor_mcts`` so self-play and reanalyze can be
    # A/B'd separately by flipping either knob.
    # Subtree reuse is N/A for reanalyze (each position is independent).
    reanalyze_use_tensor_mcts: bool = True
    # Search budget for reanalyze MCTS. None = use num_simulations (self-play
    # budget — expensive on big nets: 800 sims x 1400 games ~ hours/call).
    # Set lower (e.g. 128) to make reanalyze affordable: Gumbel keeps its
    # policy-improvement guarantee at low sim counts (m considered actions),
    # and a FRESH net at 128 sims beats a buffer-lifetime-stale 800-sim target
    # for target consistency.
    reanalyze_num_simulations: int | None = None

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

    # Policy-head architecture.
    #   flat: (default, AlphaGo-style) Conv(C->2) channel squeeze + Linear to the
    #         full action space. Fine for small/point-move action spaces.
    #   conv: (AlphaZero-style) spatial move-plane conv head — no channel squeeze,
    #         weight-shared across from-squares. For chess (4672 = 64×73) the flat
    #         head's 2-channel/128-dim squeeze is a severe bottleneck; conv removes
    #         it. Requires an 8×8-aligned latent (chess); keep "flat" for others.
    policy_head_type: str = "flat"

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

    # No-progress shuffle penalty (self-play WDL only). When a self-play game
    # draws by threefold repetition OR by the 75-move no-progress rule, its
    # (draw) value target is tilted from [0, 1, 0] to [0, 1-δ, δ] — moving δ mass
    # from Draw to Loss — teaching the value head that shuffling with no progress
    # is mildly bad. NOT applied to stalemate / insufficient-material / ply-cap
    # draws. STM-symmetric; applied to the LEGACY outcome target before the
    # selfplay_q blend. δ clamped to [0, 1]; 0.0 reproduces prior behavior. Opt-in
    # via --repetition-penalty (presets stay at 0.0).
    repetition_penalty: float = 0.0

    # Per-ply shuffle-depth weighting for the repetition penalty. The uniform δ
    # tilt blames every ply of a 3fold/no-progress game equally — including the
    # opening, which did not cause the shuffle (poor credit assignment, slow to
    # learn). When repetition_penalty_window > 0, δ is instead RAMPED by proximity
    # to the draw: full δ at the terminal (drawn) position, decaying LINEARLY to 0
    # for plies `repetition_penalty_window` or more before the end. Concretely, at
    # ply p in a game of length L (= len(observations), terminal at L-1):
    #     plies_to_end = (L - 1) - p
    #     weight       = max(0, 1 - plies_to_end / window)
    #     δ_p          = repetition_penalty * weight
    # so the per-ply decay is repetition_penalty/window of δ removed per ply moved
    # back from the draw. window=20 → full δ at the draw, 0 at 20 plies before it
    # (≈ the last 10 full moves — covers a 3fold cycle and the tail of a no-progress
    # run). 0 (default) = uniform/legacy behavior (every ply gets full δ). STM-
    # symmetric (weight depends only on ply position, not side to move).
    repetition_penalty_window: int = 0

    # Exponential variant of the per-ply weighting (preferred — matches the RL
    # discount factor, the textbook way to concentrate a terminal outcome near the
    # terminal; AlphaZero's pure-z uses γ=1 i.e. no concentration). When > 0, δ is
    # weighted by repetition_penalty_decay ** plies_to_end:
    #     plies_to_end = (L - 1) - p
    #     weight       = repetition_penalty_decay ** plies_to_end
    #     δ_p          = repetition_penalty * weight
    # Full δ at the drawn position, geometric soft-tail decay backward (no hard
    # cutoff — so a long no-progress shuffle still gets a decaying penalty over its
    # whole length, unlike the linear window). γ=0.93 → half-strength ~9.5 plies
    # (≈5 moves) back, ~0.23 at 20 plies. Takes precedence over the linear window
    # when both are set. 0.0 (default) = inactive. Clamped to [0, 1].
    repetition_penalty_decay: float = 0.0

    # Legal-move policy masking. Chess exposes a 4672-action space of which only
    # ~33 are legal at any position (99% illegal). Reference MuZero impls
    # (muzero-general, LightZero, DeepMind pseudocode) mask only at the MCTS root
    # and rely on the full-softmax CE to suppress illegal moves implicitly — fine
    # when almost every action is legal (Atari, tictactoe/connect4) but weak for
    # chess: we measured ~17% of policy mass leaking onto illegal moves at 22k.
    # When True, the policy loss keeps the standard full-softmax CE (which learns
    # legality for free — raising legal logits drains probability off illegal ones
    # through the shared softmax normalizer) and ADDS an explicit penalty on the
    # full-softmax probability mass landing on illegal moves, driving it below the
    # CE's natural floor (~17% at 22k for chess) — incl. at the latent search
    # leaves we cannot mask. We do NOT renormalize the softmax over legal moves:
    # that is shift-invariant over the legal logits and exerts no pressure to
    # raise legal above illegal, so it cannot teach legality (formally
    # full_CE = masked_CE - log P(legal) — renormalizing drops the legality term).
    # Uses the legal_actions_list already stored per ply. Default off (keeps the
    # standard path for small games); opt-in for chess via --mask-illegal-policy.
    mask_illegal_policy: bool = False
    # Weight on the illegal-mass penalty (part b above). Only active when
    # mask_illegal_policy is True.
    illegal_policy_penalty: float = 1.0

    # eval_to_wdl conversion (warmstart-only, value_head_type="wdl"): convert
    # Stockfish per-position eval (in [-1,+1] STM POV) into a soft (P_W, P_D,
    # P_L) target via parameterized 3-way logistic. ``alpha`` controls
    # sharpness (4.0 ≈ Stockfish-cp-like steepness); ``beta`` controls draw-
    # zone width. Calibrated against the Lichess real-game win-rate logistic given
    # our normalization eval=tanh(cp/800) (so eval=1 == mate, eval=0.3 == +2.5 pawns):
    # α=5.6/β=1.7 gives P_D≈0.69 at eval=0, W-L≈0.43 at eval=0.3, and P_W≈0.98 at
    # eval=1 (mate → near-certain win) — ~4× better calibration than the old 4.0/2.0
    # which under-confidenced the whole range (capped W-L at 0.88 even for mate).
    # Restores per-position teacher signal density that pure-z one-hot targets discard.
    eval_to_wdl_alpha: float = 5.6
    eval_to_wdl_beta: float = 1.7

    # --- Material / score-margin value utility (KataGo c_score analogue) ------
    # Tier-0 root-cause fix for the draw basin (see decisive_signal_plan_2026_06_23.md).
    # The self-play value target is the game OUTCOME (V^π); a weak policy draws
    # won positions, so the value head learns ≈draw everywhere → no within-
    # position resolution → MCTS has no conversion gradient. KataGo breaks this
    # by adding a bounded SCORE-MARGIN term so two winning positions are
    # rank-able. Chess analogue = MATERIAL balance. When weight > 0 and the game
    # is chess, the self-play WDL target is blended with material:
    #     m_eval  = tanh(material_margin_stm / material_value_scale)   ∈ [-1,1]
    #     target  = (1-w)·outcome_onehot + w·eval_to_wdl(m_eval)
    # material_margin_stm is read STM-relative from the observation piece planes
    # (planes 0-4 own minus 6-10 opp, weighted P/N/B/R/Q = 1/3/3/5/9; king
    # excluded). Off by default (weight 0 = exact back-compat). Warmstart
    # (external_values) targets are untouched — Stockfish already encodes material.
    material_value_weight: float = 0.0   # w (INITIAL); KataGo uses c_score≈0.5. 0 = off.
    material_value_scale: float = 5.0    # pawns mapping to ~tanh saturation (5 ≈ a rook).
    # Annealing (curriculum / shaping-decay): strong material shaping early to
    # escape the flat-target draw basin, then fade so the TRUE game outcome
    # dominates (material is a means, not the objective, in chess — unlike Go).
    # w(step) decays linearly material_value_weight → material_value_weight_final
    # over [0, material_value_anneal_frac · training_steps]. anneal_frac = 0
    # disables annealing (constant w = material_value_weight; back-compat).
    material_value_weight_final: float = 0.0
    material_value_anneal_frac: float = 0.0

    # --- Consecutive-move resignation (AlphaZero/Leela/KataGo) ----------------
    # Post-hoc, engine-agnostic: after self-play, if the side to move's stored
    # root value stays below resign_threshold for resign_consecutive of ITS OWN
    # moves, that side resigns — the game is truncated at that ply and relabeled
    # as a decisive loss for the resigning side. Protects the decisive LABEL a
    # weak policy would otherwise shuffle into a draw. Fires off the value head's
    # own estimate, so it is inert until the value head can see advantages
    # (pair with the material utility). Only threefold/no-progress-style flowing
    # games are affected in practice; off by default. NOT a compute saving (the
    # game is already played out) — label protection only.
    resign_enabled: bool = False
    resign_threshold: float = -0.9       # STM-POV root value; ≈ ≤5% expected score.
    resign_consecutive: int = 5          # consecutive own-moves below threshold.
    # AlphaZero-style calibration holdout: this fraction of games that hit the
    # resign trigger is played to its natural end instead of resigned, so the
    # false-positive rate (self_play/resign_false_positive_rate) is measurable.
    # AlphaZero played ~20% to completion; tune resign_threshold to keep FP < 5%.
    resign_holdout_frac: float = 0.15
    # Exempt SEEDED games (start_fen set) from resignation. On seeds resignation
    # is all cost, no benefit: value labels are already TB-true (all plies in-TB
    # at tb_value_weight=1.0), truncation is post-hoc (no compute saved), and it
    # fires precisely in the games where the model is about to practice the
    # conversion — deleting the finishing sequence seeding exists to generate,
    # while reducing seed/conversion to a value-calibration proxy (~95% of
    # "conversions" were resignations). With the exemption, seed/conversion and
    # seed/mate_rate become honest on-policy skill metrics.
    resign_exempt_seeded: bool = False
    # 2026-07-20 relabel policy (resignation_relabel_policy_2026_07_20.md):
    # when True, resignation NEVER overwrites a decisive natural outcome —
    # true losses keep their label AND their tail (the winner's conversion
    # demonstration, previously discarded by truncation); comeback wins keep
    # the win (anti-pessimism signal for the value head). Only oracle-free
    # DRAWS are flipped to a loss for the trigger side (+ truncated), and a
    # TB-certified drawn final position vetoes the flip (never override an
    # oracle with the value head's opinion). Trigger/FP stats are logged on
    # the FULL population (self_play/resign_trigger_*), holdout kept as the
    # historical estimator. False = legacy behavior (all triggered games
    # flipped + truncated regardless of natural outcome).
    resign_draws_only: bool = False

    # Stratified sampling: at every training batch, sample
    # floor(batch_size * warmstart_sample_frac) games from warmstart-only
    # (games with external_values populated) and the rest from self-play.
    # Maintains a permanent Stockfish-anchor in training even after the
    # buffer's FIFO eviction has flipped the in-memory composition entirely
    # to self-play. Directly attacks the catastrophic-forgetting / drawish-
    # basin failure observed in runs 0001 / 0002 (post-pool-exhaustion drift).
    # 0.0 = back-compat (no stratification). 0.3-0.5 = chess recommended.
    warmstart_sample_frac: float = 0.0
    # Anneal the warmstart anchor: warmstart_sample_frac decays linearly to
    # warmstart_sample_frac_final over [0, warmstart_anneal_frac · training_steps].
    # anneal_frac = 0 ⇒ constant (back-compat). Fade the off-distribution Stockfish
    # anchor as self-play accrues its own decisive signal (see get_warmstart_sample_frac).
    warmstart_sample_frac_final: float = 0.0
    warmstart_anneal_frac: float = 0.0

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

    # Retention "lives" for decisive games in the replay buffer. When > 1, a
    # DECISIVE game's age counts this many times slower for eviction, so it stays
    # in the buffer ~M× longer than a drawn game — raising the buffer's decisive
    # density at the RETENTION level (complements decisive_sample_frac, which acts
    # at SAMPLE time, and the warmstart anchor, which acts at GENERATION time).
    # Steady-state decisive buffer fraction with decisive inflow d:
    #   d*M / (d*M + (1-d))  → at d=0.05: M=3→14%, M=7→27%, M=14→42%.
    # 1.0 (default) = plain oldest-first FIFO. Tune down if the value overfits the
    # retained decisive set; tune up for more decisive density.
    decisive_retention_multiplier: float = 1.0

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

    # Self-attention (smolgen) backbone. Validated on the ≤5-man endgame proxy
    # (endgame_attention_findings_2026_06_28.md): attention in BOTH the representation
    # and the dynamics broke the conv tower's flat ~4% MCTS conversion plateau →
    # rising 4%→41% (KQvK 0.44→0.91) at ~equal param count — a representation win, not
    # capacity. The encoder REPLACES the conv residual tower in rep/dyn (a conv stem
    # remains). use_dyn_attention matters because the SimSiam consistency target is a
    # smolgen-rich repr latent the conv dynamics can't reproduce; matching the bodies
    # keeps geometry flowing through the MCTS rollout. pred-attention was tested and
    # did NOT help (off). NOTE: dyn-attention runs ~num_simulations/move in the search
    # hot path — a self-play throughput cost to weigh (probe before scaling depth).
    use_repr_attention: bool = False
    use_dyn_attention: bool = False
    use_pred_attention: bool = False
    use_smolgen: bool = True          # only active when an attention body is on
    attn_layers: int = 4              # encoder depth in rep & dyn (supervised proxy: L6 > L4)
    attn_heads: int = 4
    pred_attn_layers: int = 2         # only used if use_pred_attention
    # HYBRID body (strategy_2026_07_02.md §8): when >0 and an attention body is on,
    # prepend this many SE-residual conv blocks (local tactical primitives + Lc0-style
    # global channel gating) before the attention encoder in rep AND dyn. Conv's
    # data-efficient local prior for the midgame + attention's global mixing for
    # long-range geometry, at a 2x (not 10x) parameter budget.
    hybrid_stem_blocks: int = 0

    # Moves-left head (Lc0): aux categorical head predicting plies-to-game-end.
    # Breaks value saturation so search prefers faster wins over shuffling. Default
    # OFF (head not built → fully inert). support_size 10 ≈ covers ~120 plies with
    # fine resolution near 0 (where 'win faster' matters). Loss weight ~0.25 like
    # the other aux heads. The gated MCTS utility is a separate step.
    use_moves_left: bool = False
    moves_left_support_size: int = 10
    # Whether the moves-left head uses the value-style sqrt scalar_transform before
    # binning. False = LINEAR (raw plies → support), which keeps 1-ply resolution
    # across the whole range instead of compressing the long-mate (15-30 ply) region
    # into ~1.5 bins. The sqrt squash is right for value (resolution near zero) but
    # wrong for moves-left (we need resolution at large DTM). See measure_ml_accuracy.
    moves_left_use_transform: bool = True
    moves_left_loss_weight: float = 0.25
    # MLH MCTS utility (Lc0): among same-Q moves, prefer the faster win (and drag out
    # a loss). Added to the selection score as sign(-Q)*clip(slope*M, 0, max_effect),
    # gated to |Q| > threshold so it only acts in clearly won/lost positions (never
    # near draws). M = child's expected plies-to-end. Tunable via the trace A/B.
    moves_left_mcts: bool = False         # enable the search-time utility (needs a trained head)
    ml_threshold: float = 0.3             # |raw_q| above which the utility engages
    ml_slope: float = 0.005               # per-ply effect before clipping
    ml_max_effect: float = 0.1            # clip magnitude (vs value_score in [0,1])
    # Lc0 MLH utility: bonus uses (child_m − parent_m) PROGRESS, scaled by a smooth
    # |Q| ramp above ml_threshold (factor = q_const + q_linear·q' + q_square·q'^2,
    # q'=clamp((|Q|−thr)/(1−thr),0,1)). Lc0 production: thr 0.8, slope 0.0027, cap
    # 0.0345, (0, 1.6521, −0.6521). Defaults below keep the bonus a tiebreak.
    ml_q_const: float = 0.0
    ml_q_linear: float = 1.6521
    ml_q_square: float = -0.6521
    # Auxiliary-head INPUT width. The default 1-plane 1×1 projection crushes the
    # C-channel latent to a single map before the MLP, starving the head of the
    # distance-to-mate signal that lives across channels (head→DTZ corr 0.035 vs
    # MLP-on-full-latent 0.80). >1 widens it; blocks>0 adds pre-projection residual
    # blocks. value head kept at 1 (Lc0 keeps WDL value simple); widen moves-left.
    value_head_planes: int = 1
    value_head_blocks: int = 0
    moves_left_head_planes: int = 1
    moves_left_head_blocks: int = 0

    # --- Root repetition-draw penalty (terminal-aware search, the AlphaZero
    # property MuZero's latent search lacks) ----------------------------------
    # When the side to move can complete a 2nd/3rd repetition, pin that move's
    # search value to draw_score (mover POV). Side-aware automatically: a winning
    # side avoids the repeat (draw_score < its winning Q), a losing side keeps it
    # (draw_score > its losing Q) — so it preserves correct lost-side draws while
    # forcing the winning side to make progress. Repeating moves are detected from
    # the live board (GpuChessGame.repetition_move_mask) and applied at the root in
    # TensorMCTS (_select). Enabling it downgrades the triton select backend to the
    # torch path. Off by default. chess / GPU-resident self-play only.
    root_terminal_draws: bool = False
    root_terminal_draws_min_repeats: int = 2   # 2 = avoid any repeat; 3 = only block threefold
    # Extend the terminal-draw mask beyond repetition to STALEMATE + insufficient
    # material (GpuChessGame.terminal_draw_move_mask, a bounded one-ply lookahead
    # gated to <= 6-piece positions). This is the direct fix for the dominant
    # endgame failure: the winning side walking the lone king into stalemate (79%
    # of won <=5-man positions per the 2026-06-28 per-config diagnostic).
    root_terminal_draws_include_stalemate: bool = True
    # Root Syzygy tablebase probing: at each self-play ply, classify the root's
    # legal moves vs tablebases (≤ tb_max_pieces) and overwrite the root children's
    # value_score toward the DTZ-optimal conversion move (tensor_mcts._select).
    tb_root_probe: bool = False
    tb_path: str = "data/syzygy"
    tb_max_pieces: int = 5
    tb_dtz_weight: float = 0.05   # 0 = flat win value; >0 ranks winners by DTZ (shortest highest)
    # Tablebase VALUE relabeling (Lc0 rescorer analogue) — distinct from the
    # search-side bias above. When > 0, make_target blends the DTZ-shaped Syzygy
    # POSITION value (recorded per-ply during self-play in GameHistory.
    # tablebase_values) into the VALUE target at TB plies. Fixes the flat/inverted
    # value head that can't rank winning positions by distance-to-mate. 1.0 = full
    # Syzygy replacement at TB plies; keep selfplay_q_ratio≈0 so it isn't washed
    # out by the self-referential root-value blend. WDL value head only.
    tb_value_weight: float = 0.0
    # Shaping of the per-position TB value: 0 => flat WDL (win=+1), >0 => winning
    # positions ranked by their OWN DTZ (closer mate → closer +1, floored at
    # 1-shape so a win stays clearly above a draw).
    tb_value_dtz_shape: float = 0.5
    # Lc0-style HARD WDL value target at TB plies. False (default): blend the TB
    # scalar through eval_to_wdl (soft, caps W-L at ~0.88, never saturates). True:
    # one-hot win/draw/loss — saturates Q near ±1, crisp win/draw separation, and
    # activates the |Q|-gated MLH which then carries within-win distance. Pair with
    # tb_value_dtz_shape=0. Warmstart (per-position Stockfish evals) stays soft.
    tb_value_hard: bool = False
    # Moves-left (MovesLeftHead) DTM relabel — Lc0 Gaviota MLH rescoring. When > 0,
    # the moves-left target at in-TB decisive plies is replaced by |DTM| (Gaviota
    # plies-to-mate) instead of the policy-rollout plies-to-end (the shuffle length
    # in unconverted endgames). 1.0 = full replace. Needs tb_gaviota_path + a trained
    # moves-left head. This is the distance gradient the WDL value head can't supply.
    tb_moves_left_weight: float = 0.0
    tb_gaviota_path: str | None = None   # dir of Gaviota .gtb.cp4 DTM tablebases
    # Search-side policy steering (the old root-probe DTZ value bias in _select).
    # Default OFF — Lc0 rejected the DTZ policy boost; we keep self-play on-policy
    # and inject the TB signal only via the value + moves-left relabels. True restores
    # the soft search bias (needs tb_root_probe to build the prober).
    tb_steer_policy: bool = False
    # Soft TB POLICY relabel (the Lc0 DTZ policy boost — safe for us because we don't run
    # KLDGain, the only reason Lc0 disabled it; see kld_adaptive_search_analysis). When > 0,
    # blends a soft distribution over the win-PRESERVING moves into the policy TARGET at TB
    # plies: (1-w)*visits + w*tb_policy. Directly fixes the policy prior that mass-loads the
    # win-throwing move (move-selection diagnosis 2026-06-27). Annealed to *_final over
    # tb_policy_anneal_frac so the teacher fades (avoids the probe-is-a-crutch trap). Needs
    # tb_root_probe (per-move _classify). win_thresh splits winning moves from drawn/losing
    # ones; temp sharpens the DTZ-softmax among winners.
    tb_policy_weight: float = 0.0
    tb_policy_weight_final: float = 0.0
    tb_policy_anneal_frac: float = 0.0
    tb_policy_win_thresh: float = 0.5
    tb_policy_temp: float = 0.3
    # Deferred TB relabel: when steering is OFF the value/DTM/policy targets are pure
    # functions of each ply's board, so they run in ONE batched pass after the game
    # batch (over the unique in-TB FENs) instead of inline per ply — removing ~56% of
    # endgame self-play wall from the GPU hot path. >1 fans the probes across a spawn
    # process pool (each worker opens its own tablebase handles). 0/1 = single-process
    # deferred pass (still off the hot path, no parallelism). Ignored when steering on.
    tb_relabel_workers: int = 0
    # Endgame SEEDING (on-policy curriculum, gpu-resident path): a fraction of each
    # self-play round starts from tablebase endgame FENs sampled from the archive, the
    # model plays them out (no random opening; relabeled with value+DTM truth). Constant
    # fraction, NO curriculum gate (the value relabel would poison a conversion gate).
    # 0 = off. Cures covariate shift (the model trains on its OWN endgame states).
    endgame_seed_frac: float = 0.0
    endgame_seed_archive: str | None = None   # path to the seed FEN list (generate_endgame_seeds.py)
    # TB endgame ANCHOR (strategy_2026_07_02.md): inject tb_anchor_games TB-optimal
    # demonstration games (scripts/gen_tb_anchor_games.py archive) into the rolling
    # buffer every tb_anchor_interval steps, cycling the archive forever. This is the
    # validated supervised-proxy signal (KQvK 0.91) as a PERSISTENT anchor — the
    # direct TB→policy teaching channel self-play lacks. 0/None = off.
    tb_anchor_path: str | None = None
    tb_anchor_games: int = 0
    tb_anchor_interval: int = 0
    # TB ROLLOUT FILL (win adjudication by demonstration): truncate a NON-seeded
    # self-play game at its first decisive in-TB ply when the played outcome
    # contradicts the tablebase verdict, and finish it with TB-optimal play by both
    # sides. The stored game_outcome becomes the TRUE result for the WHOLE trajectory
    # (decisive z for the pre-TB middlegame plies — the channel the per-ply value
    # relabel can't reach) and the appended tail is an on-distribution conversion
    # demonstration. Seeded games are exempt (they'd be adjudicated at ply 0,
    # deleting the practice data they exist to generate).
    tb_rollout_fill: bool = False
    # MERGED self-play batch (next-run lever #11): run normal + seeded games in ONE
    # resident sweep (mixed start states, per-game opening masks) instead of
    # sequential sub-batches — removes the doubled fixed overheads and the second
    # straggler tail (~15-30% self-play wall-clock). Default off (sub-batch path).
    merged_seed_batch: bool = False
    # OPENING DIVERSITY ε-MIXTURE (next-run lever #14). When mean_plies > 0, each
    # NORMAL game opens with r ~ U[1, 2·mean] plies chosen without search:
    # with prob opening_uniform_frac a uniform-random legal move (model-INDEPENDENT
    # diversity floor — cannot collapse as the policy sharpens), else sampled from
    # the raw policy softmax at opening_policy_temp (KataGo initGamesWithPolicy —
    # plausible on-distribution branching). Opening plies store ZERO policy targets
    # (make_target's zero-CE no-op) — unlike legacy random_opening_plies, which
    # trained the policy toward its own random moves. Seeded games never open.
    opening_mix_mean_plies: int = 0
    opening_policy_temp: float = 1.5
    opening_uniform_frac: float = 0.15
    # DTM-stratified SEED CURRICULUM (reverse curriculum, Florensa 2017): sample
    # seeds whose |DTM| <= a cap that ramps seed_curriculum_dtm_easy ->
    # seed_curriculum_dtm_hard over seed_curriculum_anneal_frac of training.
    # Short mating chains first (conversion ~ p^N — benign at small N), longer
    # technique as per-move accuracy arrives. Needs the <archive>.dtm sidecar
    # (scripts/annotate_seed_dtm.py). Drawn seeds stay eligible throughout.
    seed_curriculum: bool = False
    seed_curriculum_dtm_easy: int = 8
    seed_curriculum_dtm_hard: int = 100
    seed_curriculum_anneal_frac: float = 0.5
    # D4 SYMMETRY AUGMENTATION (src/training/symmetry.py): apply a random
    # dihedral-group element to each sampled pawnless castle-free training
    # window (obs + actions + policy targets + legal masks under ONE element;
    # value/DTM/material are invariants). Anti-memorization pressure toward
    # relative-geometry features — the substrate of general endgame technique
    # (KataGo trains with Go's 8-fold symmetry the same way). Equivariance
    # verified against python-chess move generation (tests/test_symmetry.py).
    symmetry_augment: bool = False
    # Reward head width (1x1 conv planes). Historically 1 — an afterthought.
    # 2026-07-05: the dynamics-reward head is the search's in-tree mate
    # detector and THE conversion-critical component (muted-reward diag:
    # 0.20 -> 0.048 on identical weights). 8 matches value/moves-left heads.
    # Kept at 1 in presets so old checkpoints load; set per-run via CLI.
    reward_head_planes: int = 1
    # UNIFIED BUFFER (task #19): declarative batch composition. When set,
    # supersedes warmstart_sample_frac stratification. Piecewise schedule of
    # (step_fraction, {channel: weight}) with channels warmstart|anchor|selfplay;
    # weights renormalize over non-empty channels (warmup composition becomes a
    # CHOSEN ratio, not the emergent one the 2026-07-05 histogram audit found).
    batch_mixture_schedule: list | None = None
    # Per-channel cap for injected TB-anchor demonstrations in the main buffer.
    # None = legacy (anchor competes in the selfplay pool; volume emergent).
    anchor_max_size: int | None = None
    # Position sampling within a game: "per_game" (legacy; short games' plies
    # oversampled) or "per_ply" (every stored ply equally likely).
    position_sampling: str = "per_game"
    # Activation checkpointing on the attention bodies during training:
    # recompute layers in backward instead of storing activations. Exact same
    # math, ~25-30% slower train steps, memory ~n_layers-fold smaller on the
    # attention stack. Required for chess_hybrid_xl batch-512 on 32GB.
    grad_checkpoint_attention: bool = False
    # Overlap the ~141ms single-threaded sample_batch with the GPU train step via
    # a background prefetch thread (double-buffered, lock-guarded against buffer
    # writes). ~halves per-step time during the training phase. Off by default.
    prefetch_batches: bool = False
    # Auxiliary categorical head predicting the CURRENT side-to-move material
    # margin (pawns) from the LATENT state — incl. after the K dynamics unrolls,
    # so it regularizes the WORLD MODEL to preserve material through latent
    # rollouts (the meaningful chess form: final material is noisy and current
    # material is trivially in the input, but predicting it from the *latent*
    # after dynamics is not). Trained via CE on scalar_to_support(scalar_transform
    # (margin)). Queried separately (like moves-left) — does NOT touch the MCTS
    # inference tuple. chess only. Complements the value-TARGET material blend
    # above (that fixes the value label; this regularizes the representation).
    use_material_head: bool = False
    material_head_support_size: int = 8   # covers transformed material to ~±63 pawns
    material_head_loss_weight: float = 0.25   # INITIAL aux CE weight
    # Annealed on the SHARED material_value_anneal_frac timeline (init → final);
    # frac=0 ⇒ constant at the init weight. Lets the material influence fade in
    # lockstep with the value-target blend as the policy learns to convert.
    material_head_loss_weight_final: float = 0.0

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
    # Compile the NETWORK forward (recurrent_inference / predict_moves_left /
    # initial_inference) used inside MCTS, in addition to _select/_backprop. The
    # per-sim net forward otherwise runs eager (~40 conv/norm/relu kernel launches
    # each); for small models the GPU work is tiny so this CPU-side launch dispatch
    # is the self-play bottleneck (GPU underutilized, one CPU core pegged). Batch is
    # a fixed [num_parallel_games, ...] every sim → dynamic=False fuses the launches.
    tensor_mcts_compile_net: bool = False
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
            policy_head_type="conv",    # AlphaZero spatial 73-plane policy head (8×8×73=4672);
                                        # removes the flat head's 2-channel/128-dim bottleneck that
                                        # left the policy unable to rank legal moves. Chess-only.
            use_moves_left=True,        # Lc0 moves-left head: aux CE on plies-to-end. Trains during
                                        # warmstart (Stockfish game lengths). Breaks value saturation
                                        # so search can prefer faster wins over shuffling. See
                                        # mcts_audit_2026_06_17.md.
            moves_left_mcts=True,       # Enable the gated MLH search utility (Step 4). Acts only in
                                        # self-play search (post-warmup), where the head is trained.
                                        # Forces TensorMCTS off the Triton kernel onto torch _select
                                        # (the MLH/override live there; Triton port deferred).
            value_head_type="wdl",      # Lc0-style 3-output W/D/L classifier; replaces
                                        # the 5-bin scalar head. Targets game outcome z directly,
                                        # eliminating the predict-zero collapse failure mode where
                                        # the scalar head settled on the central bin.
            draw_score=-0.05,           # Anti-draw MCTS shaping (Lc0 DrawScore knob,
                                        # tree-side only). Q = WL + draw_score · D, so drawish
                                        # positions get a small negative push during search,
                                        # encouraging decisive moves over solid draws. Doesn't
                                        # affect training targets.
            root_terminal_draws=True,   # Terminal-aware search: pin root children whose move leads
                                        # to a draw to draw_score (mover-POV; winning side avoids,
                                        # losing side keeps). Covers repetition + (once the GPU mask
                                        # is extended) stalemate/insufficient. ON 2026-06-28 (was
                                        # off) per the 79%-stalemate-conversion diagnosis.
            repetition_penalty=0.35,    # Training-TARGET anti-repetition (distinct from draw_score
            repetition_penalty_decay=0.93,  # above, which is tree-only). For games drawn by 3-fold /
                                        # 50-move, rewrites the value target toward loss
                                        # ([W=0, D=1-d, L=d], d=0.35·0.93^plies_to_end). Pinned here
                                        # 2026-06-17 (was CLI-only at 0.35/0.93); default 0.0 silently
                                        # disables it if a run forgets the flag. See mcts_audit_2026_06_17.md.
            q_ratio=0.5,                # Fallback when the split knobs below are None.
            warmstart_q_ratio=0.5,      # HOT: blend 50% game outcome into the Stockfish-eval
                                        # WDL — de-saturates the value head on won positions.
                                        # Safe to run hot: the teacher signal is EXTERNAL, not
                                        # self-referential.
            selfplay_q_ratio=0.0,       # AlphaZero/Lc0 pure-outcome default. Was 0.1, but the
                                        # 2026-06-19 bug hunt showed even 0.1 DILUTES decisive
                                        # targets toward draw when root_value≈0 (the basin regime):
                                        # eval_to_wdl(0)=[.12,.76,.12], so a WIN one-hot at q=0.5
                                        # → scalar +0.48 (38% draw mass); at q=0.1 still erodes the
                                        # signal. Self-referential — keep 0.0 until the value head is
                                        # externally calibrated. (q=0.5 was why that arm collapsed fastest.)
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
            # Smolgen self-attention rep+dyn backbone — the validated endgame-conversion
            # win (see field defs above). attn_layers=4 is the throughput-friendlier
            # starting point; the supervised proxy's best was L6 (the natural scale-up
            # after a clean self-play loop confirms throughput). pred-attention stays off.
            use_repr_attention=True,
            use_dyn_attention=True,
            use_smolgen=True,
            attn_layers=4,
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
            # tensor_mcts_compile_net: SAFE local-handle torch.compile of the per-sim
            # net forward inside TensorMCTS (self._recurrent_inference etc.) — distinct
            # from compile_network below. Does NOT mutate the network's bound methods,
            # so it avoids the BatchedMCTS/reanalyze re-entry SIGILL path. ~1.4x faster
            # search; exact net-output parity; validated 2026-06-19 (cold2 runs, no SIGILL).
            tensor_mcts_compile_net=True,
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
    # chess_small: a ~5x-smaller chess net for fast hyperparameter iteration.
    # Same game (ChessGame, via game="chess") and ALL chess settings inherited;
    # only the model size shrinks, so experiments differ from the chess preset
    # only in capacity. ~22.9M -> ~4.6M params (4.9x); inference backbone (the
    # MCTS/self-play forward) 5.9M -> 1.4M (~4.3x faster turnaround). The SSL
    # projection (was 46% of params) and the inverse-dynamics head (24%) are the
    # biggest cuts. Select with `--game chess_small`; override any HP via CLI.
    configs["chess_small"] = replace(
        configs["chess"],
        hidden_planes=64,            # 128 -> 64  (block conv params scale ~quadratically)
        num_residual_blocks=6,       # 8 -> 6
        fc_hidden=64,                # 128 -> 64
        proj_hid=512, proj_out=512,  # SSL projection MLP: 1024 -> 512
        pred_hid=256, pred_out=512,  # SimSiam predictor: pred_out matches proj_out (512);
                                     # pred_hid=256 keeps the predictor bottleneck (half of proj_out)
        inverse_dynamics_hidden=96,  # 256 -> 96
        # Pure MuZero/AlphaZero exploration formulation: NO random openings — rely
        # solely on visit-count temperature (tau=1 for the first temperature_drop_step
        # plies, then temperature_final) + Dirichlet root noise. Drops the random-
        # opening double-dip the chess preset uses (random_opening_plies=8).
        random_opening_plies=0,
        # 2026-06-20: GENERATION-SCALED substrate for the data-scale convergence
        # test. Hypothesis: cold self-play stalls for lack of self-play DATA
        # VOLUME (AlphaZero/MuZero convergence is an argument about amount of
        # fresh self-play data), not because the target has zero resolution. So
        # scale GENERATION (wide parallel self-play) and hold the replay reuse at
        # the paper's ~1:1 (train ~one sample per generated sample), keeping it
        # pure self-play (no oracle / win-adjudication) per the paper's intent.
        #
        # Per-round 1:1 identity:  batch_size * self_play_interval
        #                          == num_self_play_games * avg_game_len(~165).
        num_parallel_games=384,      # 256 -> 384: per-SWEEP width, bounded by GPU
                                     #   memory. Resident self-play holds the whole
                                     #   sweep's trajectories on-GPU, so peak mem ~
                                     #   num_parallel_games * max_plies (+ end-of-sweep
                                     #   stack spike). 512 @ max_plies=750 OOM'd
                                     #   STOCHASTICALLY on sweeps whose deepest game
                                     #   hit ~740-750 plies. 384*750=288k < the
                                     #   512*680=348k sweep that already fit at 23.4GB,
                                     #   so even a full-750-ply sweep is safe here.
                                     #   num_self_play_games runs as multiple sweeps.
        num_self_play_games=1024,    # 2048 -> 1024 (2026-06-23): the 1:1 step:sample
                                     #   ratio didn't help materially, so generate less
                                     #   (self-play is the wall-clock bottleneck) and let
                                     #   reuse rise. ~3 sweeps/round.
        self_play_interval=660,      # held at 660: with 1024 games, 512*660 / (1024*165)
                                     #   -> replay reuse ~2.0 (each fresh sample trained ~2x).
        batch_size=512,              # 256 -> 512: modest; lowered hard from the 4096
                                     #   mis-step. With 1:1 the per-step decisive count
                                     #   still rises because the buffer is decisive-rich
                                     #   early. (Total games over a run = steps*batch/L,
                                     #   so 'more games' comes from steps, not batch.)
        replay_buffer_size=5120,     # 1500 -> 5120 (~2.5 rounds resident), 3.4x the
                                     #   baseline. Affordable only with SPARSE policies
                                     #   (Path B, 2026-06-20): dense was ~10MB/game ->
                                     #   8192=82GB OOM-kill; sparse ~3.2MB/game. Capped
                                     #   at 5120 (not 8192) because the 60GB cgroup limit
                                     #   binds on the self-play-ROUND peak = resting
                                     #   buffer + round spike (new games + ~12GB dense
                                     #   per-sweep CPU transient + base). Measured 30.9GB
                                     #   RSS at buf=2048 -> 8192 would peak ~58GB; 5120
                                     #   peaks ~46GB (anon, not cache -> OOM-relevant).
        reanalyze_batch_size=1400,   # 256 -> 1400: scaled commensurately with the
                                     #   8192 buffer (keeps the original 256/1500 ~=17%
                                     #   per-call refresh fraction). Reanalyze runs MCTS
                                     #   on EVERY position of each sampled game, so this
                                     #   adds ~30-40% search wall-clock — dial down if
                                     #   too slow. reanalyze_interval stays 1024.
        decisive_sample_frac=0.0,    # AXED: stratified decisive oversampling overfit
                                     #   a collapsing ~5-13 game pool (2026-06-19
                                     #   verdict smoking-gun #1). 0.0 -> flat PER.
        # NOTE: selfplay_q_ratio is already 0.0 (inherited from chess) — the
        # self-referential Q-blend that "collapses fastest" stays off.
        # --- Cold-start substrate fixes (2026-06-23) ---------------------------
        repetition_penalty=0.0,      # 0.35 -> 0: it's a late-regime lever (inert until
                                     #   the policy can convert) AND a draw-amplifier
                                     #   imprint (06-19 verdict: teaches "how soon the draw
                                     #   comes," not board value). A confound for the cold
                                     #   material experiment — re-enable once converting.
        value_head_init_std=0.01,    # 0.0 -> 0.01: zero-init blocks the value head's body
                                     #   gradient at cold start; 0.01 lets it flow without
                                     #   dominating (cold-start ablation baseline).
        max_plies=750,               # 1024 -> 750: match the 384-parallel GPU-memory
                                     #   envelope (384*750=288k < the fitted 512*680 sweep;
                                     #   1024 would exceed the tested size and risk OOM).
    )
    # chess_hybrid (2026-07-03, strategy doc §8): the ~2x HYBRID — conv-SE stem +
    # attention body in rep AND dyn (matched bodies per the EXP-0 lesson), a shared
    # 1-layer attention prediction body before the heads split, and widened value /
    # moves-left projections (the 1-plane heads provably under-read the latent).
    # d128 with 4 heads fixes the head-dim-16 thinness of chess_small's d64 attention.
    # Rationale: conv's data-efficient local prior for the midgame (the conv-vs-attn
    # production gap at matched steps) + attention's global mixing for endgame
    # geometry (the 4%-vs-41% supervised proxy gap), at a size the current data
    # budget can feed. Aux SSL/inverse-dynamics heads grow with d128 (training-only);
    # slim them further if VRAM binds.
    configs["chess_hybrid"] = replace(
        configs["chess_small"],
        hidden_planes=128,           # attention d_model; 4 heads x head-dim 32
        fc_hidden=128,
        use_repr_attention=True,
        use_dyn_attention=True,
        use_smolgen=True,
        attn_layers=4,
        attn_heads=4,
        hybrid_stem_blocks=2,        # conv-SE stem depth in rep & dyn
        use_pred_attention=True,     # shared pre-head attention body (PredictionNetwork);
                                     #   feeds policy + value + moves-left (material stays
                                     #   on the raw latent — it's a world-model regularizer)
        pred_attn_layers=2,          # ~12 MFLOPs/layer vs ~100 for one dynamics apply:
                                     #   +~10%/sim per layer — depth 2 is affordable
        value_head_planes=8,         # 1 -> 8: un-bottleneck the value projection
        moves_left_head_planes=8,    # 1 -> 8: latent holds DTZ at corr 0.80, the
                                     #   1-plane head reads it at 0.035
    )
    # chess_hybrid_xl (2026-07-06, strategy §16): the 5x SCALE TEST — same shape
    # as chess_hybrid, 4.9x the inference params (23.97M vs 4.85M). d288 with 8
    # heads (head-dim 36), 6-layer bodies, 3-layer prediction body, 3-block
    # conv-SE stems, fc 256. Paired with 2x schedule (30k warmstart / 300k
    # total). Reward head planes stay 8 (set per-run via CLI). Board-game
    # scaling prior (Jones 2021): expect left-shifted milestones (sample
    # efficiency), overfit risk on the unchanged 5120 buffer — watch
    # SF-agreement as the tripwire.
    configs["chess_hybrid_xl"] = replace(
        configs["chess_hybrid"],
        hidden_planes=288,
        fc_hidden=256,
        attn_layers=6,
        attn_heads=8,
        hybrid_stem_blocks=3,
        pred_attn_layers=3,
    )
    return configs.get(game, configs["tictactoe"])
