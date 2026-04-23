"""LR scheduler wiring: warmup + MultiStepLR composition in the trainer.

Run with: pytest tests/test_lr_schedule.py
"""

import pytest
import torch

from src.config import MuZeroConfig
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork
from src.training.trainer import MuZeroTrainer


def _trainer(tmp_path, **cfg_overrides):
    """Build a minimal trainer with a throwaway tictactoe net."""
    base_kwargs = dict(
        game="tictactoe",
        hidden_planes=16,
        num_residual_blocks=1,
        latent_h=3, latent_w=3,
        fc_hidden=16,
        batch_size=4,
        training_steps=1000,
        replay_buffer_size=32,
        min_buffer_size=4,
        num_self_play_games=4,
        self_play_interval=100,
        lr=1e-3,
        checkpoint_interval=10_000,
        eval_interval=10_000,
    )
    base_kwargs.update(cfg_overrides)
    cfg = MuZeroConfig(**base_kwargs)
    game = TicTacToe()
    net = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks,
        latent_h=cfg.latent_h,
        latent_w=cfg.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
    )
    return MuZeroTrainer(
        cfg, game, net, run_id="lr-test", device="cpu",
        log_dir=str(tmp_path / "runs"),
        checkpoints_dir=str(tmp_path / "ckpts"),
    )


def _current_lr(t: MuZeroTrainer) -> float:
    return t.optimizer.param_groups[0]["lr"]


def test_no_warmup_starts_at_config_lr(tmp_path):
    """lr_warmup_steps=0 → scheduler is plain MultiStepLR; lr == config.lr at step 0."""
    t = _trainer(tmp_path, lr=2e-3, lr_warmup_steps=0, lr_decay_milestones=[])
    assert _current_lr(t) == pytest.approx(2e-3)


def test_warmup_ramps_linearly_to_config_lr(tmp_path):
    """500-step linear warmup: lr starts tiny, hits config.lr at step lr_warmup_steps."""
    t = _trainer(
        tmp_path, lr=2e-3, lr_warmup_steps=500, lr_decay_milestones=[],
    )

    lr0 = _current_lr(t)
    assert lr0 < 2e-3 * 0.01, f"expected tiny start lr, got {lr0}"

    # Halfway through warmup.
    for _ in range(250):
        t.scheduler.step()
    lr_mid = _current_lr(t)
    assert 2e-3 * 0.4 < lr_mid < 2e-3 * 0.6, (
        f"expected ~half config.lr at mid-warmup, got {lr_mid}"
    )

    # End of warmup (500 total steps).
    for _ in range(250):
        t.scheduler.step()
    assert _current_lr(t) == pytest.approx(2e-3, rel=1e-3)

    # Past warmup: still config.lr (no decay milestones).
    for _ in range(100):
        t.scheduler.step()
    assert _current_lr(t) == pytest.approx(2e-3, rel=1e-3)


def test_warmup_then_decay_milestones_fire(tmp_path):
    """After warmup, MultiStepLR milestones still take effect."""
    t = _trainer(
        tmp_path,
        lr=2e-3,
        lr_warmup_steps=50,
        training_steps=1000,
        lr_decay_milestones=[0.5],  # at step 500
        lr_decay_factor=0.1,
    )
    # Warmup done at step 50.
    for _ in range(50):
        t.scheduler.step()
    assert _current_lr(t) == pytest.approx(2e-3, rel=1e-3)

    # Milestone is at step 500 in the MultiStepLR's internal counter.
    # SequentialLR resets the sub-scheduler counter when it hands off, so
    # the fire point in cumulative terms is lr_warmup_steps + 500 = 550.
    for _ in range(499):
        t.scheduler.step()
    assert _current_lr(t) == pytest.approx(2e-3, rel=1e-3), (
        "milestone must not have fired yet"
    )
    t.scheduler.step()  # cumulative step 550 — decay fires
    assert _current_lr(t) == pytest.approx(2e-4, rel=1e-3), (
        f"expected 10× decay, got {_current_lr(t)}"
    )


def test_chess_preset_has_warmup_and_doubled_lr():
    """Sanity: the chess preset actually ships with lr=2e-3 and 500-step warmup."""
    from src.config import get_config
    cfg = get_config("chess")
    assert cfg.lr == pytest.approx(2e-3), f"expected 2e-3, got {cfg.lr}"
    assert cfg.lr_warmup_steps == 500, f"expected 500, got {cfg.lr_warmup_steps}"
    assert cfg.use_gumbel is False, "chess should default to PUCT, not Gumbel"
