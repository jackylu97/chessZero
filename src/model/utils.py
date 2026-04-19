"""Utility building blocks for neural networks."""

import torch
import torch.nn as nn


def norm_layer(num_channels: int, spatial_shape: tuple[int, int]) -> nn.Module:
    """LayerNorm over (channels, height, width) per sample.

    Why LayerNorm and not BatchNorm: BN maintains running_mean/running_var EMAs
    that assume IID batch statistics. MuZero's dynamics is applied recurrently
    during training unroll (K=5 applications per sample), violating IID — one
    outlier batch can pull running_var by 30-80x and the net takes hundreds of
    steps to recover. Observed on a chess training run (checkpoint diff:
    dynamics BN running_var jumped 0.3 -> 17 across a single loss-spike event).
    LayerNorm has no running stats, so the failure mode disappears.

    Choice of LayerNorm vs GroupNorm: both fix the running-stats issue. LN is
    the canonical choice in MuZero derivatives — DeepMind's updated architecture
    (post-2020) and LightZero (default: norm_type='LN') both use LN. We match.

    Shape convention: nn.LayerNorm([C, H, W]) normalizes across all C*H*W
    values per sample. This requires knowing spatial dims at construction time,
    which is fine since each game's board size is fixed in config.
    """
    return nn.LayerNorm([num_channels, *spatial_shape], eps=1e-5)


class ResidualBlock(nn.Module):
    """Post-activation residual block: Conv -> Norm -> ReLU -> Conv -> Norm, add, ReLU.

    Matches the AlphaZero-era residual block ordering (ResNet v1, He 2015) used
    by both muzero-general and LightZero. The spatial_shape arg is required
    because norm_layer uses nn.LayerNorm, which needs per-(C,H,W) affine params.
    """

    def __init__(self, planes: int, spatial_shape: tuple[int, int]):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn1 = norm_layer(planes, spatial_shape)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes, spatial_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


def mlp_head(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
    """Simple MLP: Linear-ReLU-Linear."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


# --- Categorical support encoding (MuZero paper Appendix F) ---

def scalar_transform(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """Invertible value transform h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x.

    Compresses large values while preserving sign. Makes learning
    easier when values span a wide range.
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def inverse_scalar_transform(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """Inverse of scalar_transform: recovers original scale."""
    # Solve h(x) = y for x, where h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x
    # Approximation via the inverse of the dominant sqrt term
    return torch.sign(x) * (
        ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert scalar values to categorical distribution over support.

    Maps a scalar x to a probability distribution over the discrete support
    {-support_size, ..., 0, ..., support_size} using linear interpolation
    between the two nearest support points.

    Args:
        x: (B,) or (B, 1) scalar values
        support_size: half-width of support (total bins = 2*support_size + 1)

    Returns:
        (B, 2*support_size + 1) probability distribution
    """
    x = x.reshape(-1)
    # Clamp to support range
    x = x.clamp(-support_size, support_size)

    num_bins = 2 * support_size + 1
    floor = x.floor().long()
    frac = x - x.floor()

    # Indices into the support vector (shifted so -support_size -> index 0)
    low_idx = (floor + support_size).clamp(0, num_bins - 1)
    high_idx = (low_idx + 1).clamp(0, num_bins - 1)

    probs = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
    probs.scatter_(1, low_idx.unsqueeze(1), (1 - frac).unsqueeze(1))
    probs.scatter_(1, high_idx.unsqueeze(1), frac.unsqueeze(1))

    return probs


def support_to_scalar(probs: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert categorical distribution back to scalar value.

    Args:
        probs: (B, 2*support_size + 1) logits (will be softmaxed)
        support_size: half-width of support

    Returns:
        (B,) scalar values
    """
    # Apply softmax to get probabilities from logits
    probs = torch.softmax(probs, dim=1)
    support = torch.arange(-support_size, support_size + 1,
                           device=probs.device, dtype=probs.dtype)
    return (probs * support).sum(dim=1)
