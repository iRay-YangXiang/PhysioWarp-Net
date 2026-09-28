"""Regularization for order-preserving temporal transformations."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def temporal_monotonicity_loss(
    acquisition_times: torch.Tensor,
    delta_seconds: torch.Tensor,
    *,
    min_interval_seconds: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize transformed acquisition times that are too close or reversed.

    The transformed time is ``phi = acquisition_times + delta_seconds`` using
    the project-wide convention ``source_time = output_time + delta``.

    Args:
        acquisition_times: Strictly increasing times in seconds with shape
            ``[B, T]`` or ``[T]``.
        delta_seconds: Displacement field in seconds with shape
            ``[B, T, Z, H, W]``.
        min_interval_seconds: Required minimum increase in ``phi`` between
            adjacent output frames.

    Returns:
        A tuple ``(loss, fraction_of_violations)`` of scalar tensors. The loss
        is the mean ``relu(min_interval_seconds - diff(phi))``.
    """
    if delta_seconds.ndim != 5:
        raise ValueError("delta_seconds must have shape [B, T, Z, H, W]")
    if not delta_seconds.is_floating_point():
        raise TypeError("delta_seconds must be floating point")
    if min_interval_seconds < 0:
        raise ValueError("min_interval_seconds must be non-negative")

    batch_size, num_times = delta_seconds.shape[:2]
    if num_times < 2:
        raise ValueError("delta_seconds must contain at least two time points")
    if acquisition_times.ndim == 1:
        acquisition_times = acquisition_times.unsqueeze(0).expand(batch_size, -1)
    if acquisition_times.ndim != 2 or acquisition_times.shape != (
        batch_size,
        num_times,
    ):
        raise ValueError("acquisition_times must have shape [T] or [B, T]")
    if not acquisition_times.is_floating_point():
        raise TypeError("acquisition_times must be floating point")
    if acquisition_times.device != delta_seconds.device:
        raise ValueError(
            "acquisition_times and delta_seconds must be on the same device"
        )
    if not torch.isfinite(acquisition_times).all():
        raise ValueError("acquisition_times must contain only finite values")
    if not torch.isfinite(delta_seconds).all():
        raise ValueError("delta_seconds must contain only finite values")
    if torch.any(acquisition_times[:, 1:] <= acquisition_times[:, :-1]):
        raise ValueError("acquisition_times must be strictly increasing")

    times = acquisition_times.to(dtype=delta_seconds.dtype)
    phi = times[:, :, None, None, None] + delta_seconds
    transformed_intervals = phi[:, 1:] - phi[:, :-1]
    violations = transformed_intervals < min_interval_seconds
    loss = functional.relu(
        min_interval_seconds - transformed_intervals
    ).mean()
    violation_fraction = violations.to(delta_seconds.dtype).mean()
    return loss, violation_fraction
