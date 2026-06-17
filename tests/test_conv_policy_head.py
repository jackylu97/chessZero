"""Tests for the AlphaZero-style ConvPolicyHead (chess spatial policy head).

The critical correctness property is the index ORDERING: the conv emits
(B, move_planes, H, W) and must flatten so that logit index ==
from_sq * move_planes + move_type, matching chess.py::_move_to_action
(from_sq * 73 + move_type). If the permute/reshape is wrong, the policy head
would be trained against transposed targets — silently learning garbage.
"""

import pytest
import torch

from src.model.muzero_net import ConvPolicyHead, MuZeroNetwork


def test_ordering_matches_action_encoding():
    """flat[from_sq*P + move_type] must equal the spatial logit at
    (move_type, from_row, from_col)."""
    torch.manual_seed(0)
    C, H, W, A = 8, 8, 8, 4672
    head = ConvPolicyHead(hidden_planes=C, action_space_size=A, latent_h=H, latent_w=W)
    assert head.move_planes == 73
    # The output conv is zero-init; randomize it so the ordering is observable.
    torch.nn.init.normal_(head.proj.weight, std=0.5)
    torch.nn.init.normal_(head.proj.bias, std=0.5)

    x = torch.randn(2, C, H, W)
    flat = head(x)
    assert flat.shape == (2, A)
    spatial = head.proj(torch.relu(head.norm(head.mix(x))))  # (2, 73, 8, 8)
    P = head.move_planes
    for from_sq in (0, 1, 8, 27, 63):
        row, col = divmod(from_sq, 8)
        for move_type in (0, 1, 55, 56, 64, 72):
            idx = from_sq * P + move_type
            assert torch.allclose(flat[:, idx], spatial[:, move_type, row, col], atol=1e-5), \
                f"index mismatch at from_sq={from_sq}, move_type={move_type}"


def test_zero_init_gives_uniform_policy():
    """Default (un-randomized) head zero-inits its output conv -> all-zero
    logits -> uniform policy at start (MuZero stability trick)."""
    head = ConvPolicyHead(8, 4672, 8, 8)
    out = head(torch.randn(3, 8, 8, 8))
    assert torch.allclose(out, torch.zeros_like(out))


def test_rejects_indivisible_action_space():
    with pytest.raises(ValueError, match="divisible"):
        ConvPolicyHead(8, 4673, 8, 8)  # 4673 % 64 != 0


def test_network_builds_and_forwards_with_conv_head():
    net = MuZeroNetwork(
        observation_channels=8, action_space_size=4672, hidden_planes=8,
        num_blocks=1, latent_h=8, latent_w=8, input_h=8, input_w=8,
        fc_hidden=16, value_support_size=1, reward_support_size=1,
        value_head_type="wdl", policy_head_type="conv",
    )
    assert isinstance(net.prediction.policy_head, ConvPolicyHead)
    policy_logits, _value = net.prediction(torch.randn(2, 8, 8, 8))
    assert policy_logits.shape == (2, 4672)


def test_flat_head_still_default():
    """Other games / default keep the flat head (Sequential)."""
    net = MuZeroNetwork(
        observation_channels=8, action_space_size=4672, hidden_planes=8,
        num_blocks=1, latent_h=8, latent_w=8, input_h=8, input_w=8,
        fc_hidden=16, value_support_size=1, reward_support_size=1,
    )
    assert isinstance(net.prediction.policy_head, torch.nn.Sequential)
