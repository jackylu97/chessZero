"""Tests for the inverse-dynamics aux loss (ICM/Pathak) + value-head init flag.

The inverse-dynamics head predicts a_k from (h_k, h_{k+1}); it is the validated
fix for action-blind dynamics (the EfficientZero consistency loss alone does not
fix it — see scripts/probe_fix_candidates.py and the dynamics_gradient_starvation
memory note). Key property: it delivers gradient to the world-model body (and the
action embedding) from training step 0, bypassing the zero-init prediction heads.

Run with: pytest tests/test_inverse_dynamics.py
"""

import numpy as np
import torch

from src.config import MuZeroConfig
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork, InverseDynamicsHead
from src.training.replay_buffer import GameHistory
from src.training.trainer import MuZeroTrainer


# --- InverseDynamicsHead module contract -----------------------------------

def test_inverse_head_output_shape_and_grad():
    head = InverseDynamicsHead(hidden_planes=8, latent_h=3, latent_w=3,
                               action_space_size=9, hidden=16)
    h_k = torch.randn(4, 8, 3, 3, requires_grad=True)
    h_next = torch.randn(4, 8, 3, 3, requires_grad=True)
    logits = head(h_k, h_next)
    assert logits.shape == (4, 9)
    logits.sum().backward()
    # Gradient flows into BOTH hidden states (no detach).
    assert h_k.grad is not None and torch.isfinite(h_k.grad).all()
    assert h_next.grad is not None and torch.isfinite(h_next.grad).all()


def test_network_builds_inverse_head_only_when_flagged():
    common = dict(observation_channels=3, action_space_size=9, hidden_planes=8,
                  num_blocks=1, latent_h=3, latent_w=3, input_h=3, input_w=3, fc_hidden=16,
                  value_support_size=2, reward_support_size=1)
    assert MuZeroNetwork(**common, use_inverse_dynamics_loss=False).inverse_dynamics_head is None
    net = MuZeroNetwork(**common, use_inverse_dynamics_loss=True, inverse_dynamics_hidden=16)
    assert isinstance(net.inverse_dynamics_head, InverseDynamicsHead)


def test_value_head_init_std_makes_output_nonzero():
    """value_head_init_std>0 → value head produces nonzero output at init, so the
    head's Jacobian w.r.t. its input is nonzero and body gradient is unblocked."""
    common = dict(observation_channels=3, action_space_size=9, hidden_planes=8,
                  num_blocks=1, latent_h=3, latent_w=3, input_h=3, input_w=3, fc_hidden=16,
                  value_support_size=2, reward_support_size=1, value_head_type="wdl")
    obs = torch.randn(4, 3, 3, 3)
    zero_net = MuZeroNetwork(**common, value_head_init_std=0.0)
    _, _, v0 = zero_net.initial_inference_logits(obs)
    assert torch.allclose(v0, torch.zeros_like(v0)), "zero-init value head should output 0"
    small_net = MuZeroNetwork(**common, value_head_init_std=0.1)
    _, _, v1 = small_net.initial_inference_logits(obs)
    assert not torch.allclose(v1, torch.zeros_like(v1)), "small-init value head should be nonzero"


# --- End-to-end _train_step ------------------------------------------------

def _make_game_history(length=6, action_space=9, n_planes=3, hist=2):
    game = GameHistory()
    for i in range(length):
        # n_planes-channel per-ply obs, distinct per step.
        game.observations.append(torch.full((n_planes, 3, 3), float(i)))
        game.actions.append(i % action_space)
        game.rewards.append(0.0)
        game.policies.append(np.full(action_space, 1 / action_space, dtype=np.float32))
        game.root_values.append(0.0)
    game.game_outcome = 0.0
    game.legal_actions_list = [list(range(action_space)) for _ in range(length)]
    return game


def test_train_step_with_inverse_and_single_frame_trains_body_from_step0():
    """Full _train_step with inverse dynamics + single-frame consistency +
    small value-head init: finite losses, 'inverse_loss' reported, and the
    representation/dynamics body + action embedding receive gradient at step 0
    even though the value head is (small-)init near zero."""
    torch.manual_seed(0); np.random.seed(0)
    HIST = 2
    game = TicTacToe()
    NP = game.num_planes

    config = MuZeroConfig(
        game="tictactoe", hidden_planes=8, num_residual_blocks=1,
        latent_h=3, latent_w=3, fc_hidden=16, value_support_size=2, reward_support_size=1,
        num_unroll_steps=3, batch_size=4, min_buffer_size=2, use_amp=False,
        history_frames=HIST,
        use_consistency_loss=True, consistency_single_frame_target=True,
        consistency_loss_weight=2.0, proj_hid=16, proj_out=16, pred_hid=8, pred_out=16,
        use_inverse_dynamics_loss=True, inverse_dynamics_loss_weight=1.0,
        inverse_dynamics_hidden=16, value_head_init_std=0.01,
    )
    network = MuZeroNetwork(
        observation_channels=NP * HIST, action_space_size=game.action_space_size,
        hidden_planes=8, num_blocks=1, latent_h=3, latent_w=3,
        input_h=game.board_size[0], input_w=game.board_size[1], fc_hidden=16,
        value_support_size=2, reward_support_size=1,
        use_consistency_loss=True, proj_hid=16, proj_out=16, pred_hid=8, pred_out=16,
        value_head_init_std=0.01, use_inverse_dynamics_loss=True, inverse_dynamics_hidden=16,
    )
    trainer = MuZeroTrainer(
        config, game, network, run_id="test_inverse",
        device="cpu", log_dir="/tmp/test_inverse_runs", checkpoints_dir="/tmp/test_inverse_ckpts",
    )
    for _ in range(4):
        trainer.replay_buffer.save_game(_make_game_history(length=6, n_planes=NP, hist=HIST))

    # zero grads, run one step, inspect grads BEFORE the optimizer overwrites them
    # by checking param.grad right after backward is impossible here, so instead
    # confirm params moved (gradient flowed) after the step.
    repr_before = {n: p.detach().clone() for n, p in network.representation.named_parameters()}
    emb_before = network.dynamics.action_embedding.weight.detach().clone()
    inv_before = {n: p.detach().clone() for n, p in network.inverse_dynamics_head.named_parameters()}

    loss_info = trainer._train_step()

    assert "inverse_loss" in loss_info
    assert np.isfinite(loss_info["inverse_loss"])
    assert np.isfinite(loss_info["total_loss"])
    assert np.isfinite(loss_info["consistency_loss"])

    changed_repr = any(not torch.equal(p, repr_before[n])
                       for n, p in network.representation.named_parameters())
    changed_emb = not torch.equal(network.dynamics.action_embedding.weight, emb_before)
    changed_inv = any(not torch.equal(p, inv_before[n])
                      for n, p in network.inverse_dynamics_head.named_parameters())
    assert changed_repr, "representation body did not train — body gradient still blocked"
    assert changed_emb, "action embedding did not train — inverse loss not reaching it"
    assert changed_inv, "inverse-dynamics head did not train"


def test_train_step_without_inverse_omits_key_and_works():
    """use_inverse_dynamics_loss=False: no inverse head, step still runs, and the
    reported inverse_loss is exactly 0."""
    torch.manual_seed(0); np.random.seed(0)
    game = TicTacToe()
    config = MuZeroConfig(
        game="tictactoe", hidden_planes=8, num_residual_blocks=1,
        latent_h=3, latent_w=3, fc_hidden=16, value_support_size=2, reward_support_size=1,
        num_unroll_steps=3, batch_size=4, min_buffer_size=2, use_amp=False,
        use_consistency_loss=False, use_inverse_dynamics_loss=False,
    )
    network = MuZeroNetwork(
        observation_channels=game.num_planes, action_space_size=game.action_space_size,
        hidden_planes=8, num_blocks=1, latent_h=3, latent_w=3,
        input_h=game.board_size[0], input_w=game.board_size[1], fc_hidden=16,
        value_support_size=2, reward_support_size=1,
        use_inverse_dynamics_loss=False,
    )
    trainer = MuZeroTrainer(
        config, game, network, run_id="test_no_inverse",
        device="cpu", log_dir="/tmp/test_inverse_runs", checkpoints_dir="/tmp/test_inverse_ckpts",
    )
    for _ in range(4):
        trainer.replay_buffer.save_game(_make_game_history(length=6, n_planes=game.num_planes, hist=1))
    loss_info = trainer._train_step()
    assert network.inverse_dynamics_head is None
    assert np.isfinite(loss_info["total_loss"])
    assert loss_info["inverse_loss"] == 0.0
