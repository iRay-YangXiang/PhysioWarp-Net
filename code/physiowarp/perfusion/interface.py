"""Interface for quantitative perfusion models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class PerfusionModel(nn.Module, ABC):
    """Abstract differentiable mapping from dynamic CT to quantitative maps."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        acquisition_times: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map ``x [B,T,1,Z,H,W]`` and times to maps ``[B,Z,H,W]``."""
        raise NotImplementedError
