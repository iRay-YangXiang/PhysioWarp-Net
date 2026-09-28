"""Image fidelity objectives."""

from __future__ import annotations

import torch


def l1_image_loss(original: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
    """Return mean absolute error for matching ``[B,T,C,Z,H,W]`` images."""
    if original.ndim != 6 or warped.shape != original.shape:
        raise ValueError("original and warped must share shape [B, T, C, Z, H, W]")
    return torch.mean(torch.abs(original - warped))
