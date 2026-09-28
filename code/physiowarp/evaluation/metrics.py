"""Metrics for temporal deformation quality and failure modes."""

from __future__ import annotations

import torch


def deformation_diagnostics(
    acquisition_times: torch.Tensor,
    delta: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    delta_max_seconds: float,
    saturation_tolerance: float = 0.01,
    min_interval_seconds: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Measure displacement saturation, clipping, and temporal ordering.

    Args:
        acquisition_times: Physical times ``[B,T]`` or ``[T]``.
        delta: Effective displacement ``[B,T,Z,H,W]`` in seconds.
        support_mask: Spatial support ``[B,1,Z,H,W]`` or ``[B,Z,H,W]``.
        delta_max_seconds: Configured absolute displacement bound.
        saturation_tolerance: Relative distance from the bound counted as
            saturation. A value of 0.01 marks ``|delta| >= 0.99 * max``.
        min_interval_seconds: Minimum desired adjacent transformed interval.

    Returns:
        Scalar tensor metrics. Fractions are computed only inside support.
    """
    if delta.ndim != 5:
        raise ValueError("delta must have shape [B, T, Z, H, W]")
    if support_mask.ndim == 5 and support_mask.shape[1] == 1:
        support_mask = support_mask[:, 0]
    if support_mask.shape != (delta.shape[0], *delta.shape[2:]):
        raise ValueError("support_mask must have shape [B,1,Z,H,W] or [B,Z,H,W]")
    if delta_max_seconds <= 0:
        raise ValueError("delta_max_seconds must be positive")
    if not 0 <= saturation_tolerance < 1:
        raise ValueError("saturation_tolerance must lie in [0, 1)")
    if min_interval_seconds < 0:
        raise ValueError("min_interval_seconds must be non-negative")

    batch_size, num_times = delta.shape[:2]
    if acquisition_times.ndim == 1:
        acquisition_times = acquisition_times.unsqueeze(0).expand(batch_size, -1)
    if acquisition_times.shape != (batch_size, num_times):
        raise ValueError("acquisition_times must have shape [T] or [B,T]")
    if acquisition_times.device != delta.device or support_mask.device != delta.device:
        raise ValueError("all inputs must be on the same device")

    times = acquisition_times.to(dtype=delta.dtype)
    support = support_mask.to(dtype=delta.dtype)[:, None].expand_as(delta)
    support_count = support.sum().clamp_min(1.0)
    absolute_delta = delta.abs()
    saturation_threshold = delta_max_seconds * (1.0 - saturation_tolerance)
    saturation_fraction = (
        (absolute_delta >= saturation_threshold).to(delta.dtype) * support
    ).sum() / support_count

    query_times = times[:, :, None, None, None] + delta
    below = query_times < times[:, :1, None, None, None]
    above = query_times > times[:, -1:, None, None, None]
    clipping_fraction = ((below | above).to(delta.dtype) * support).sum() / support_count

    phi_intervals = query_times[:, 1:] - query_times[:, :-1]
    interval_support = support[:, 1:]
    interval_count = interval_support.sum().clamp_min(1.0)
    violation_fraction = (
        (phi_intervals < min_interval_seconds).to(delta.dtype) * interval_support
    ).sum() / interval_count

    outside = 1.0 - support
    return {
        "mean_abs_delta_inside_seconds": (absolute_delta * support).sum()
        / support_count,
        "max_abs_delta_outside_seconds": (absolute_delta * outside).amax(),
        "delta_saturation_fraction": saturation_fraction,
        "temporal_clipping_fraction": clipping_fraction,
        "monotonicity_violation_fraction": violation_fraction,
    }
