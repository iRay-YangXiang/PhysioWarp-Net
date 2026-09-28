"""Small differentiable quantitative surrogate for synthetic experiments."""

from __future__ import annotations

import torch

from .interface import PerfusionModel


class SoftTemporalSummaryPerfusion(PerfusionModel):
    """Estimate soft peak time and a smooth peak-amplitude proxy.

    This module is a test surrogate, not a clinical CTP implementation.
    Both outputs have shape ``[B, Z, H, W]``.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        acquisition_times: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return differentiable ``tmax`` and ``rcbf`` proxy maps."""
        if x.ndim != 6 or x.shape[2] != 1:
            raise ValueError("x must have shape [B, T, 1, Z, H, W]")
        batch_size, num_times = x.shape[:2]
        if acquisition_times.ndim == 1:
            acquisition_times = acquisition_times.unsqueeze(0).expand(batch_size, -1)
        if acquisition_times.shape != (batch_size, num_times):
            raise ValueError("acquisition_times must have shape [T] or [B, T]")
        if acquisition_times.device != x.device:
            raise ValueError("x and acquisition_times must be on the same device")

        curve = x[:, :, 0]
        peak_weights = torch.softmax(curve / self.temperature, dim=1)
        times = acquisition_times.to(dtype=x.dtype)[:, :, None, None, None]
        tmax = (peak_weights * times).sum(dim=1)
        smooth_peak = self.temperature * torch.logsumexp(
            curve / self.temperature,
            dim=1,
        )
        rcbf = torch.sigmoid(smooth_peak)
        return {"rcbf": rcbf, "tmax": tmax}
