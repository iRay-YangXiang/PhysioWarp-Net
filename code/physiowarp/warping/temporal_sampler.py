"""Differentiable interpolation along the temporal dimension only."""

from __future__ import annotations

import torch


def temporal_warp(
    x: torch.Tensor,
    acquisition_times: torch.Tensor,
    delta_seconds: torch.Tensor,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample a dynamic volume at displaced physical times.

    The sign convention is ``source_time = output_time + delta``. Therefore,
    positive displacement samples a later point on the original temporal
    curve. Interpolation changes no spatial coordinate.

    Args:
        x: Dynamic image with shape ``[B, T, C, Z, H, W]``.
        acquisition_times: Strictly increasing physical times in seconds with
            shape ``[B, T]`` or ``[T]``.
        delta_seconds: Temporal displacement in seconds with shape
            ``[B, T, Z, H, W]`` or ``[B, T, 1, Z, H, W]``.
        padding_mode: Boundary behavior. Only ``"border"`` is currently
            supported and clamps queries to the valid time range.

    Returns:
        Temporally warped image with shape ``[B, T, C, Z, H, W]``.
    """
    if padding_mode != "border":
        raise ValueError('padding_mode must be "border"')
    if x.ndim != 6:
        raise ValueError("x must have shape [B, T, C, Z, H, W]")
    if not x.is_floating_point():
        raise TypeError("x must be floating point")

    batch_size, num_times, num_channels, depth, height, width = x.shape
    if num_times < 2:
        raise ValueError("x must contain at least two time frames")

    if acquisition_times.ndim == 1:
        acquisition_times = acquisition_times.unsqueeze(0).expand(batch_size, -1)
    if acquisition_times.ndim != 2 or acquisition_times.shape != (
        batch_size,
        num_times,
    ):
        raise ValueError("acquisition_times must have shape [T] or [B, T]")
    if not acquisition_times.is_floating_point():
        raise TypeError("acquisition_times must be floating point")
    if acquisition_times.device != x.device:
        raise ValueError("x and acquisition_times must be on the same device")
    if not torch.isfinite(acquisition_times).all():
        raise ValueError("acquisition_times must contain only finite values")
    if torch.any(acquisition_times[:, 1:] <= acquisition_times[:, :-1]):
        raise ValueError("acquisition_times must be strictly increasing")

    if delta_seconds.ndim == 6:
        if delta_seconds.shape[2] != 1:
            raise ValueError("the channel dimension of delta_seconds must be 1")
        delta_seconds = delta_seconds.squeeze(2)
    expected_delta_shape = (batch_size, num_times, depth, height, width)
    if delta_seconds.ndim != 5 or delta_seconds.shape != expected_delta_shape:
        raise ValueError(
            "delta_seconds must have shape [B, T, Z, H, W] "
            "or [B, T, 1, Z, H, W]"
        )
    if not delta_seconds.is_floating_point():
        raise TypeError("delta_seconds must be floating point")
    if delta_seconds.device != x.device:
        raise ValueError("x and delta_seconds must be on the same device")
    if not torch.isfinite(delta_seconds).all():
        raise ValueError("delta_seconds must contain only finite values")

    times = acquisition_times.to(dtype=x.dtype)
    delta = delta_seconds.to(dtype=x.dtype)
    spatial_size = depth * height * width

    output_times = times[:, :, None, None, None]
    query_times = output_times + delta
    query_times = torch.clamp(
        query_times,
        min=times[:, :1, None, None, None],
        max=times[:, -1:, None, None, None],
    )
    flat_queries = query_times.reshape(batch_size, num_times * spatial_size).contiguous()

    upper_indices = torch.searchsorted(times.contiguous(), flat_queries, right=True)
    upper_indices = upper_indices.clamp(min=1, max=num_times - 1)
    lower_indices = upper_indices - 1

    lower_times = torch.gather(times, 1, lower_indices)
    upper_times = torch.gather(times, 1, upper_indices)
    denominator = (upper_times - lower_times).clamp_min(
        torch.finfo(x.dtype).eps
    )
    weights = (flat_queries - lower_times) / denominator
    weights = weights.reshape(batch_size, num_times, spatial_size)
    weights = weights.permute(0, 2, 1).unsqueeze(1)

    temporal_last = x.permute(0, 2, 3, 4, 5, 1).reshape(
        batch_size,
        num_channels,
        spatial_size,
        num_times,
    )
    gather_shape = (batch_size, num_channels, spatial_size, num_times)
    lower_gather = lower_indices.reshape(batch_size, num_times, spatial_size)
    lower_gather = lower_gather.permute(0, 2, 1).unsqueeze(1).expand(gather_shape)
    upper_gather = upper_indices.reshape(batch_size, num_times, spatial_size)
    upper_gather = upper_gather.permute(0, 2, 1).unsqueeze(1).expand(gather_shape)

    lower_values = torch.gather(temporal_last, -1, lower_gather)
    upper_values = torch.gather(temporal_last, -1, upper_gather)
    warped = lower_values + weights * (upper_values - lower_values)
    return warped.permute(0, 3, 1, 2).reshape(
        batch_size,
        num_times,
        num_channels,
        depth,
        height,
        width,
    )
