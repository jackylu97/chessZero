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
