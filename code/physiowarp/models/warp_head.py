"""Prediction head for temporal basis coefficient maps."""

from __future__ import annotations

import torch
from torch import nn


class WarpHead(nn.Module):
    """Map aggregated spatial features to ``K`` coefficient maps."""

    def __init__(self, in_channels: int, num_basis: int) -> None:
        super().__init__()
        if in_channels < 1 or num_basis < 1:
            raise ValueError("channel counts and num_basis must be positive")
        hidden_channels = max(in_channels // 2, num_basis)
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, num_basis, kernel_size=1),
        )
        final_layer = self.layers[-1]
        assert isinstance(final_layer, nn.Conv3d)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return coefficients ``[B, K, Z, H, W]`` from 3D features."""
        if features.ndim != 5:
            raise ValueError("features must have shape [B, F, Z, H, W]")
        return self.layers(features)
