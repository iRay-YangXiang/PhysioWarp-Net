"""Tests for the differentiable synthetic perfusion surrogate."""

from __future__ import annotations

import torch

from physiowarp.perfusion import SoftTemporalSummaryPerfusion


def test_surrogate_shapes_and_gradient() -> None:
    x = torch.randn(2, 5, 1, 2, 3, 3, requires_grad=True)
    times = torch.tensor([0.0, 1.0, 2.5, 4.0, 6.0])

    maps = SoftTemporalSummaryPerfusion()(x, times)
    (maps["tmax"].mean() + maps["rcbf"].mean()).backward()

    assert maps["tmax"].shape == (2, 2, 3, 3)
    assert maps["rcbf"].shape == (2, 2, 3, 3)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()