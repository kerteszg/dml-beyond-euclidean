"""Small encoders for the MWE (ambient Euclidean features only).

Head maps (ℓ₂ / expmap0) are applied outside the network so the same encoder
class can serve all three geometries.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """MLP for the synthetic Gaussian-leaf probe (32 -> 64 -> 16)."""

    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallConvEncoder(nn.Module):
    """Tiny CNN for CIFAR-10 (3ch, 32x32) and Omniglot (1ch, 28x28).

    Three conv blocks -> global average pool -> linear embedding.
    Kept small so CPU defaults finish in minutes.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.head(h)
