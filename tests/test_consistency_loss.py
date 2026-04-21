"""Tests for EfficientZero consistency loss (Ye 2021).

SimSiam-style self-supervision: rolled-out dynamics latent must match
representation(future_obs), measured via negative cosine similarity. Follows
LightZero's lzero/policy/efficientzero.py / efficientzero_model.py.

Run with: pytest tests/test_consistency_loss.py
"""

import numpy as np
import pytest
import torch

from src.config import MuZeroConfig
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork, PredictionHead, ProjectionNetwork
from src.training.replay_buffer import GameHistory, ReplayBuffer
from src.training.trainer import MuZeroTrainer, _negative_cosine_similarity


# --- Projection + predictor module contracts -----------------------------

def test_projection_output_shape():
    """ProjectionNetwork: (B, in_dim) → (B, out_dim)."""
    proj = ProjectionNetwork(in_dim=64, hidden=32, out_dim=16)
    x = torch.randn(8, 64)
    y = proj(x)
    assert y.shape == (8, 16)


def test_prediction_head_output_shape():
    """PredictionHead: (B, in_dim) → (B, out_dim)."""
    pred = PredictionHead(in_dim=16, hidden=8, out_dim=16)
    x = torch.randn(4, 16)
    y = pred(x)
    assert y.shape == (4, 16)


# --- MuZeroNetwork.project() -----------------------------------------------

def _make_network(use_consistency=True, proj_hid=32, proj_out=32, pred_hid=16, pred_out=32):
    return MuZeroNetwork(
        observation_channels=3,
        action_space_size=9,
        hidden_planes=8,
        num_blocks=1,
        latent_h=3, latent_w=3,
        input_h=3, input_w=3,
        fc_hidden=16,
        value_support_size=2,
        reward_support_size=1,
        use_consistency_loss=use_consistency,
        proj_hid=proj_hid, proj_out=proj_out,
        pred_hid=pred_hid, pred_out=pred_out,
    )


def test_project_with_grad_returns_predictor_output():
    """with_grad=True → predictor(projection(hidden)); shape (B, pred_out); grad flows."""
    net = _make_network(proj_out=32, pred_out=32)
    hidden = torch.randn(4, 8, 3, 3, requires_grad=True)
    out = net.project(hidden, with_grad=True)
    assert out.shape == (4, 32)
    assert out.requires_grad
    out.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_project_without_grad_returns_detached_projection():
    """with_grad=False → projection(hidden).detach(); shape (B, proj_out); no grad."""
    net = _make_network(proj_out=32, pred_out=64)
    hidden = torch.randn(4, 8, 3, 3)
    out = net.project(hidden, with_grad=False)
    assert out.shape == (4, 32)  # proj_out, not pred_out
    assert not out.requires_grad


def test_project_flattens_spatial_latent():
    """Projection input is flattened (B, C*H*W) — works across latent shapes."""
    net = _make_network()
    hidden = torch.randn(2, 8, 3, 3)
    out = net.project(hidden, with_grad=True)
    assert out.shape[0] == 2


def test_project_raises_when_consistency_loss_disabled():
    """project() asserts use_consistency_loss=True at construction."""
    net = _make_network(use_consistency=False)
    hidden = torch.randn(2, 8, 3, 3)
    with pytest.raises(AssertionError):
        net.project(hidden, with_grad=True)


# --- Negative cosine similarity -----------------------------------------

def test_negative_cosine_similarity_range():
    """Output per sample in [-1, 1] (negative of cos similarity)."""
    p = torch.randn(16, 32)
    z = torch.randn(16, 32)
    sim = _negative_cosine_similarity(p, z)
    assert sim.shape == (16,)
    assert torch.all(sim >= -1 - 1e-5)
    assert torch.all(sim <= 1 + 1e-5)


def test_negative_cosine_similarity_identical_vectors_is_minus_one():
    """-cos(p, p) = -1 exactly (vectors aligned)."""
    p = torch.randn(4, 8)
    sim = _negative_cosine_similarity(p, p.clone())
    torch.testing.assert_close(sim, torch.full((4,), -1.0), atol=1e-5, rtol=1e-5)


def test_negative_cosine_similarity_handles_zero_vectors():
    """Zero-norm input → F.normalize eps guard → finite output (0 similarity)."""
    p = torch.zeros(2, 8)
    z = torch.randn(2, 8)
    sim = _negative_cosine_similarity(p, z)
    assert torch.isfinite(sim).all()


# --- make_target returns K+1 obs + mask ----------------------------------

def _make_game_history(length: int = 6, action_space: int = 9):
    game = GameHistory()
    for i in range(length):
        game.observations.append(torch.full((3, 3, 3), float(i)))  # distinct per step
        game.actions.append(i % action_space)
        game.rewards.append(0.0)
        game.policies.append(np.full(action_space, 1 / action_space, dtype=np.float32))
        game.root_values.append(0.0)
    game.game_outcome = 0.0
    return game


def test_make_target_returns_k_plus_one_obs():
    """obs_list has K+1 entries at state_index..state_index+K."""
    game = _make_game_history(length=10)
    obs_list, mask, actions, *_ = game.make_target(
        state_index=2, num_unroll_steps=5, td_steps=-1, discount=1.0, action_space_size=9
    )
    assert len(obs_list) == 6
    assert len(mask) == 6
    assert len(actions) == 5
    # Each obs is distinct — all indices within the game.
    for i, obs in enumerate(obs_list):
        assert obs[0, 0, 0].item() == pytest.approx(2 + i)
    assert mask == [1.0] * 6


def test_make_target_masks_past_game_end():
    """Indices past len(game) have mask=0.0 and reuse last valid obs."""
    game = _make_game_history(length=4)
    obs_list, mask, *_ = game.make_target(
        state_index=2, num_unroll_steps=5, td_steps=-1, discount=1.0, action_space_size=9
    )
    # Positions 2, 3 are valid (indices < 4); 4, 5, 6, 7 are past end.
    assert mask == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # Past-end obs reuse last valid obs (index 3).
    for i in (2, 3, 4, 5):
        assert obs_list[i][0, 0, 0].item() == pytest.approx(3.0)


def test_make_target_root_obs_matches_state_index():
    """obs_list[0] is the observation at state_index (unchanged from pre-EZ behavior)."""
    game = _make_game_history(length=6)
    obs_list, *_ = game.make_target(
        state_index=3, num_unroll_steps=5, td_steps=-1, discount=1.0, action_space_size=9
    )
    assert obs_list[0][0, 0, 0].item() == pytest.approx(3.0)


def test_sample_batch_stacks_target_observations():
    """ReplayBuffer.sample_batch returns (B, K+1, C, H, W) + (B, K+1) mask."""
    buf = ReplayBuffer(max_size=10)
    for _ in range(3):
        buf.save_game(_make_game_history(length=5))
    batch, _, _ = buf.sample_batch(
        batch_size=4, num_unroll_steps=5, td_steps=-1, discount=1.0, action_space_size=9
    )
    assert "target_observations" in batch
    assert "target_obs_mask" in batch
    assert batch["target_observations"].shape == (4, 6, 3, 3, 3)
    assert batch["target_obs_mask"].shape == (4, 6)
    # Back-compat: observations[:, 0] still present as root obs.
    assert batch["observations"].shape == (4, 3, 3, 3)


# --- End-to-end _train_step with consistency loss -------------------------

def test_train_step_with_consistency_loss_produces_finite_loss():
    """A training step with use_consistency_loss=True:
    - runs without NaN/Inf,
    - returns a 'consistency_loss' key in loss_info,
    - updates parameters (nonzero grad through projection/predictor).
    """
    torch.manual_seed(0)
    np.random.seed(0)

    config = MuZeroConfig(
        game="tictactoe",
        hidden_planes=8,
        num_residual_blocks=1,
        latent_h=3, latent_w=3,
        fc_hidden=16,
        value_support_size=2,
        reward_support_size=1,
        num_unroll_steps=3,
        batch_size=4,
        num_self_play_games=2,
        training_steps=10,
        min_buffer_size=2,
        use_amp=False,
        use_consistency_loss=True,
        consistency_loss_weight=2.0,
        proj_hid=16, proj_out=16, pred_hid=8, pred_out=16,
    )

    game = TicTacToe()
    network = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h, latent_w=config.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_consistency_loss=True,
        proj_hid=config.proj_hid, proj_out=config.proj_out,
        pred_hid=config.pred_hid, pred_out=config.pred_out,
    )

    trainer = MuZeroTrainer(
        config, game, network, run_id="test_consistency",
        device="cpu", log_dir="/tmp/test_consistency_runs",
        checkpoints_dir="/tmp/test_consistency_ckpts",
    )

    # Seed the buffer with a few games so sample_batch can draw.
    for _ in range(4):
        gh = _make_game_history(length=6)
        gh.legal_actions_list = [list(range(9)) for _ in range(len(gh))]
        trainer.replay_buffer.save_game(gh)

    # Capture projection + predictor params before training.
    proj_before = {n: p.detach().clone() for n, p in network.projection.named_parameters()}
    pred_before = {n: p.detach().clone() for n, p in network.prediction_head.named_parameters()}

    loss_info = trainer._train_step()

    assert "consistency_loss" in loss_info
    assert np.isfinite(loss_info["consistency_loss"])
    assert np.isfinite(loss_info["total_loss"])

    # At least one param in the predictor must have changed — confirms the
    # consistency loss contributed gradient.
    changed_pred = any(
        not torch.equal(p, pred_before[n]) for n, p in network.prediction_head.named_parameters()
    )
    changed_proj = any(
        not torch.equal(p, proj_before[n]) for n, p in network.projection.named_parameters()
    )
    assert changed_pred, "predictor params unchanged — consistency loss didn't backprop"
    assert changed_proj, "projection params unchanged — consistency loss didn't backprop"


def test_train_step_without_consistency_loss_still_works():
    """Back-compat: use_consistency_loss=False skips the consistency branch
    and produces the same loss keys (minus consistency_loss=0)."""
    torch.manual_seed(0)
    np.random.seed(0)

    config = MuZeroConfig(
        game="tictactoe",
        hidden_planes=8, num_residual_blocks=1,
        latent_h=3, latent_w=3, fc_hidden=16,
        value_support_size=2, reward_support_size=1,
        num_unroll_steps=3, batch_size=4, num_self_play_games=2,
        training_steps=10, min_buffer_size=2,
        use_amp=False,
        use_consistency_loss=False,
    )
    game = TicTacToe()
    network = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h, latent_w=config.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_consistency_loss=False,
    )
    trainer = MuZeroTrainer(
        config, game, network, run_id="test_no_consistency",
        device="cpu", log_dir="/tmp/test_no_consistency_runs",
        checkpoints_dir="/tmp/test_no_consistency_ckpts",
    )
    for _ in range(4):
        gh = _make_game_history(length=6)
        gh.legal_actions_list = [list(range(9)) for _ in range(len(gh))]
        trainer.replay_buffer.save_game(gh)

    loss_info = trainer._train_step()
    assert np.isfinite(loss_info["total_loss"])
    # Key is present but value is 0 (consistency_loss tensor remained zeros).
    assert loss_info["consistency_loss"] == pytest.approx(0.0)
