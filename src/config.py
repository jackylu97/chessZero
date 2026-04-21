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
    value_loss_weight: float = 0.25  # downweight value loss (Reanalyze paper)
    num_unroll_steps: int = 5  # K: number of future steps to unroll
    td_steps: int = -1  # -1 means use full game return
    discount: float = 1.0  # 1.0 for board games (no discounting)
    training_steps: int = 5000
    checkpoint_interval: int = 500
    log_interval: int = 50

    # Replay buffer
    replay_buffer_size: int = 5000  # number of games
    min_buffer_size: int = 50  # min games before training starts

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
            num_simulations=200,     # 200 for ~12 visits/candidate on m=16 Gumbel (was 100 ≈ 6/candidate)
            batch_size=256,
            training_steps=100000,
            checkpoint_interval=1000,
            lr_decay_milestones=[0.5, 0.75],  # decay 10× at 50k and 75k
            replay_buffer_size=10000,   # bumped from 1000 so Stockfish warmstart games
                                        # aren't evicted before they've trained the net
            min_buffer_size=500,
            num_self_play_games=100,
            self_play_interval=1000,   # 2:1 train:selfplay ratio
            lr=5e-4,
            dirichlet_alpha=0.03,
            td_steps=-1,             # full MC return; bootstrapping poisons targets while value head is broken
            temperature_drop_step=30,
            reanalyze_interval=1000,   # keep 1:1 with self_play_interval
            reanalyze_batch_size=100,
            num_parallel_games=128,    # bumped from 64 — batched-sync run_batch fits 24GB
            sample_k=50,               # Sampled MuZero: sample K distinct actions per node (Hubert 2021 Proposed Modification)
            eval_interval=5000,
            use_consistency_loss=True, # EfficientZero SimSiam consistency loss on dynamics rollouts
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
