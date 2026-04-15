"""Utility building blocks for neural networks."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard residual block: Conv-BN-ReLU-Conv-BN + skip connection."""

    def __init__(self, planes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

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
