"""Shared spatial feature extraction for dynamic CT frames."""

from __future__ import annotations

import torch
from torch import nn


class SpatialEncoder(nn.Module):
    """Extract full-resolution spatial features from individual 3D frames.

    Input has shape ``[N, C, Z, H, W]`` and output has shape
    ``[N, F, Z, H, W]``.
    """

    def __init__(self, in_channels: int = 1, feature_channels: int = 16) -> None:
        super().__init__()
        if in_channels < 1 or feature_channels < 1:
            raise ValueError("channel counts must be positive")
        num_groups = min(4, feature_channels)
        while feature_channels % num_groups != 0:
            num_groups -= 1
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, feature_channels),
            nn.SiLU(),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, feature_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return features for frames ``x`` with shape ``[N, C, Z, H, W]``."""
        if x.ndim != 5:
            raise ValueError("x must have shape [N, C, Z, H, W]")
        return self.layers(x)
