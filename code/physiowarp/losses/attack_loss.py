"""Smooth threshold-crossing objectives for quantitative maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

AttackMode = Literal["rcbf_down", "rcbf_up", "tmax_up", "tmax_down"]


@dataclass(frozen=True)
class AttackLossConfig:
    """Threshold and smoothness settings for quantitative attacks."""

    mode: AttackMode = "tmax_up"
    rcbf_threshold: float = 0.30
    tmax_threshold_seconds: float = 6.0
    sigmoid_sharpness: float = 10.0
    baseline_margin_rcbf: float = 0.03
    baseline_margin_tmax: float = 0.5


class ThresholdAttackLoss(nn.Module):
    """Encourage a quantitative map to cross a configured threshold."""

    def __init__(self, config: AttackLossConfig = AttackLossConfig()) -> None:
        super().__init__()
        if config.sigmoid_sharpness <= 0:
            raise ValueError("sigmoid_sharpness must be positive")
        if config.baseline_margin_rcbf < 0 or config.baseline_margin_tmax < 0:
            raise ValueError("baseline margins must be non-negative")
        self.config = config

    @staticmethod
    def _map_shape(mask: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        if mask.ndim == 5 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.shape != reference.shape:
            raise ValueError(f"{name} must match map shape [B, Z, H, W]")
        return mask.to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        baseline_map: torch.Tensor,
        attacked_map: torch.Tensor,
        candidate_mask: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return loss and success metrics for maps shaped ``[B,Z,H,W]``."""
        if baseline_map.ndim != 4 or attacked_map.shape != baseline_map.shape:
            raise ValueError("baseline_map and attacked_map must share [B,Z,H,W]")
        if not baseline_map.is_floating_point() or not attacked_map.is_floating_point():
            raise TypeError("quantitative maps must be floating point")
        if baseline_map.device != attacked_map.device:
            raise ValueError("baseline_map and attacked_map must share a device")
        candidate = self._map_shape(candidate_mask, baseline_map, "candidate_mask")
        support = self._map_shape(support_mask, baseline_map, "support_mask")
        if torch.any(candidate < 0) or torch.any(support < 0):
            raise ValueError("candidate and support masks must be non-negative")

        mode = self.config.mode
        if mode.startswith("rcbf"):
            threshold = self.config.rcbf_threshold
            margin = self.config.baseline_margin_rcbf
        else:
            threshold = self.config.tmax_threshold_seconds
            margin = self.config.baseline_margin_tmax

        increasing = mode in ("rcbf_up", "tmax_up")
        if increasing:
            baseline_eligible = baseline_map < threshold - margin
            soft_score = torch.sigmoid(
                self.config.sigmoid_sharpness * (attacked_map - threshold)
            )
            hard_success = attacked_map >= threshold
        else:
            baseline_eligible = baseline_map > threshold + margin
            soft_score = torch.sigmoid(
                self.config.sigmoid_sharpness * (threshold - attacked_map)
            )
            hard_success = attacked_map <= threshold

        weights = candidate * support * baseline_eligible.to(baseline_map.dtype)
        valid = baseline_eligible & (candidate > 0) & (support > 0)
        num_candidates = valid.sum().to(dtype=baseline_map.dtype)
        weight_sum = weights.sum()
        has_candidates = weight_sum > 0
        safe_denominator = weight_sum.clamp_min(torch.finfo(baseline_map.dtype).eps)
        soft_success = (soft_score * weights).sum() / safe_denominator
        soft_success = torch.where(
            has_candidates,
            soft_success,
            attacked_map.sum() * 0.0,
        )
        loss = torch.where(has_candidates, 1.0 - soft_success, soft_success)
        hard_fraction = (
            (hard_success & valid).to(baseline_map.dtype).sum()
            / num_candidates.clamp_min(1.0)
        )
        return {
            "loss": loss,
            "soft_success": soft_success,
            "hard_success_fraction": hard_fraction,
            "num_candidate_voxels": num_candidates,
        }
