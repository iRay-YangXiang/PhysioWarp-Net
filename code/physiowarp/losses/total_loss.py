"""Composite objectives for staged training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physiowarp.warping.monotonicity import temporal_monotonicity_loss

from .image_loss import l1_image_loss
from .temporal_loss import (
    first_order_delta_loss,
    max_abs_delta,
    mean_abs_delta,
    second_order_delta_loss,
)


@dataclass(frozen=True)
class LossWeights:
    """Configurable scalar weights for the Stage 1 objective."""

    attack: float = 1.0
    image: float = 1.0
    temporal_first: float = 0.1
    temporal_second: float = 0.1
    monotonicity: float = 1.0
    delta_size: float = 0.1


def stage1_total_loss(
    original: torch.Tensor,
    warped: torch.Tensor,
    acquisition_times: torch.Tensor,
    delta_effective: torch.Tensor,
    attack_loss: torch.Tensor,
    *,
    fixed_mask: torch.Tensor | None = None,
    weights: LossWeights = LossWeights(),
    min_interval_seconds: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Combine and report all Stage 1 losses and deformation metrics."""
    image = l1_image_loss(original, warped)
    temporal_first = first_order_delta_loss(delta_effective, fixed_mask)
    temporal_second = second_order_delta_loss(delta_effective, fixed_mask)
    monotonicity, violation_fraction = temporal_monotonicity_loss(
        acquisition_times,
        delta_effective,
        min_interval_seconds=min_interval_seconds,
    )
    delta_mean = mean_abs_delta(delta_effective, fixed_mask)
    delta_max = max_abs_delta(delta_effective, fixed_mask)
    total = (
        weights.attack * attack_loss
        + weights.image * image
        + weights.temporal_first * temporal_first
        + weights.temporal_second * temporal_second
        + weights.monotonicity * monotonicity
        + weights.delta_size * delta_mean
    )
    return {
        "total_loss": total,
        "attack_loss": attack_loss,
        "image_loss": image,
        "delta_first_order": temporal_first,
        "delta_second_order": temporal_second,
        "monotonicity_loss": monotonicity,
        "monotonicity_violation_fraction": violation_fraction,
        "mean_abs_delta_seconds": delta_mean,
        "max_abs_delta_seconds": delta_max,
    }
