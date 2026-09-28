"""Stage-wise PhysioWarp-Net model definitions."""

from __future__ import annotations

import torch
from torch import nn

from physiowarp.warping.temporal_basis import TemporalBSplineBasis
from physiowarp.warping.temporal_sampler import temporal_warp

from .spatial_encoder import SpatialEncoder
from .warp_head import WarpHead


class Stage1TemporalWarpNet(nn.Module):
    """Predict smooth temporal deformation inside a fixed spatial ROI.

    Args:
        num_basis: Number of temporal B-spline coefficient maps.
        delta_max_seconds: Absolute displacement bound in physical seconds.
        feature_channels: Number of shared spatial feature channels.
        spline_degree: Degree of the temporal B-spline basis.

    Input:
        x: Dynamic image ``[B, T, 1, Z, H, W]``.
        acquisition_times: Physical times ``[B, T]`` or ``[T]``.
        fixed_mask: ROI weights ``[B, 1, Z, H, W]`` in ``[0, 1]``.
    """

    def __init__(
        self,
        num_basis: int = 4,
        delta_max_seconds: float = 1.0,
        feature_channels: int = 16,
        spline_degree: int = 3,
    ) -> None:
        super().__init__()
        if delta_max_seconds <= 0:
            raise ValueError("delta_max_seconds must be positive")
        self.delta_max_seconds = delta_max_seconds
        self.spatial_encoder = SpatialEncoder(1, feature_channels)
        self.warp_head = WarpHead(feature_channels * 3, num_basis)
        self.temporal_basis = TemporalBSplineBasis(num_basis, spline_degree)

    def forward(
        self,
        x: torch.Tensor,
        acquisition_times: torch.Tensor,
        fixed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Warp ``x`` using learned smooth displacement inside ``fixed_mask``.

        Returns ``x_warped [B,T,1,Z,H,W]``, ``coefficients [B,K,Z,H,W]``,
        ``delta_raw``, ``delta``, ``delta_effective``, and ``phi`` each with
        temporal/spatial shape ``[B,T,Z,H,W]`` except where stated otherwise.
        """
        if x.ndim != 6:
            raise ValueError("x must have shape [B, T, 1, Z, H, W]")
        batch_size, num_times, num_channels, depth, height, width = x.shape
        if num_channels != 1:
            raise ValueError("Stage 1 requires a single image channel")
        if fixed_mask.shape != (batch_size, 1, depth, height, width):
            raise ValueError("fixed_mask must have shape [B, 1, Z, H, W]")
        if fixed_mask.device != x.device:
            raise ValueError("x and fixed_mask must be on the same device")
        if not fixed_mask.is_floating_point():
            raise TypeError("fixed_mask must be floating point")
        if not torch.isfinite(fixed_mask).all():
            raise ValueError("fixed_mask must contain only finite values")
        if torch.any((fixed_mask < 0) | (fixed_mask > 1)):
            raise ValueError("fixed_mask values must lie in [0, 1]")

        spatial_input = x.reshape(
            batch_size * num_times,
            num_channels,
            depth,
            height,
            width,
        )
        frame_features = self.spatial_encoder(spatial_input).reshape(
            batch_size,
            num_times,
            -1,
            depth,
            height,
            width,
        )
        temporal_features = torch.cat(
            (
                frame_features.mean(dim=1),
                frame_features.std(dim=1, correction=0),
                frame_features.amax(dim=1),
            ),
            dim=1,
        )
        coefficients = self.warp_head(temporal_features)
        basis = self.temporal_basis(acquisition_times)
        if basis.shape[:2] == (1, num_times) and batch_size != 1:
            basis = basis.expand(batch_size, -1, -1)
        if basis.shape[:2] != (batch_size, num_times):
            raise ValueError("acquisition_times must have shape [T] or [B, T]")

        delta_raw = torch.einsum("bkzhw,btk->btzhw", coefficients, basis)
        delta = self.delta_max_seconds * torch.tanh(delta_raw)
        delta_effective = delta * fixed_mask[:, None, 0]

        times = acquisition_times
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(batch_size, -1)
        phi = times.to(dtype=delta.dtype)[:, :, None, None, None] + delta_effective
        x_warped = temporal_warp(x, acquisition_times, delta_effective)
        return {
            "x_warped": x_warped,
            "coefficients": coefficients,
            "delta_raw": delta_raw,
            "delta": delta,
            "delta_effective": delta_effective,
            "phi": phi,
        }
