"""Regularization and metrics for temporal displacement fields."""

from __future__ import annotations

import torch


def _validate_delta(delta: torch.Tensor) -> None:
    if delta.ndim != 5:
        raise ValueError("delta must have shape [B, T, Z, H, W]")
    if not delta.is_floating_point():
        raise TypeError("delta must be floating point")


def _spatial_weights(delta: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask[:, 0]
    expected_shape = (delta.shape[0], *delta.shape[2:])
    if mask.ndim != 4 or mask.shape != expected_shape:
        raise ValueError("mask must have shape [B, 1, Z, H, W] or [B, Z, H, W]")
    if mask.device != delta.device:
        raise ValueError("delta and mask must be on the same device")
    return mask.to(dtype=delta.dtype).unsqueeze(1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    expanded_weights = weights.expand_as(values)
    denominator = expanded_weights.sum().clamp_min(torch.finfo(values.dtype).eps)
    return (values * expanded_weights).sum() / denominator


def first_order_delta_loss(
    delta: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return masked mean ``|delta[t+1] - delta[t]|`` for ``[B,T,Z,H,W]``."""
    _validate_delta(delta)
    if delta.shape[1] < 2:
        raise ValueError("first-order loss requires at least two time points")
    return _weighted_mean(
        torch.abs(delta[:, 1:] - delta[:, :-1]),
        _spatial_weights(delta, mask),
    )


def second_order_delta_loss(
    delta: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return masked mean temporal second difference for ``[B,T,Z,H,W]``."""
    _validate_delta(delta)
    if delta.shape[1] < 3:
        raise ValueError("second-order loss requires at least three time points")
    second_difference = delta[:, 2:] - 2.0 * delta[:, 1:-1] + delta[:, :-2]
    return _weighted_mean(
        torch.abs(second_difference),
        _spatial_weights(delta, mask),
    )


def mean_abs_delta(
    delta: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return masked mean absolute displacement in seconds."""
    _validate_delta(delta)
    return _weighted_mean(torch.abs(delta), _spatial_weights(delta, mask))


def max_abs_delta(
    delta: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return maximum absolute displacement, optionally after mask weighting."""
    _validate_delta(delta)
    values = torch.abs(delta)
    weights = _spatial_weights(delta, mask)
    if weights is not None:
        values = values * weights
    return values.amax()
