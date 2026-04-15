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
    reward_support_size: int = 1  # scalar reward for board games

    # MCTS
    num_simulations: int = 25
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_drop_step: int = 15  # move number to switch to final temp

    # Training
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
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

    # AMP
    use_amp: bool = True

    # Evaluation
    eval_games: int = 50
    eval_interval: int = 500

    # Device
    device: str = "cuda"

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
            num_simulations=25,
            batch_size=64,
            training_steps=5000,
            replay_buffer_size=5000,
            min_buffer_size=50,
            num_self_play_games=100,
            self_play_interval=100,
            lr=1e-3,
            dirichlet_alpha=0.3,
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
            num_simulations=200,
            batch_size=256,
            training_steps=500000,
            replay_buffer_size=100000,
            min_buffer_size=500,
            num_self_play_games=100,
            self_play_interval=500,
            lr=2e-4,
            dirichlet_alpha=0.03,
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
