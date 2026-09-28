"""Tests for the fixed-ROI Stage 1 temporal warp network."""

from __future__ import annotations

import torch

from physiowarp.models import Stage1TemporalWarpNet


def test_stage1_shapes_identity_initialization_and_roi() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 5, 1, 2, 4, 4)
    times = torch.tensor([0.0, 0.8, 2.0, 3.5, 5.0])
    mask = torch.zeros(2, 1, 2, 4, 4)
    mask[:, :, :, 1:3, 1:3] = 1.0
    model = Stage1TemporalWarpNet(num_basis=4, feature_channels=4)

    output = model(x, times, mask)

    assert output["coefficients"].shape == (2, 4, 2, 4, 4)
    assert output["delta"].shape == (2, 5, 2, 4, 4)
    assert output["phi"].shape == (2, 5, 2, 4, 4)
    torch.testing.assert_close(output["x_warped"], x)
    torch.testing.assert_close(
        output["delta_effective"] * (1.0 - mask[:, None, 0]),
        torch.zeros_like(output["delta_effective"]),
    )


def test_stage1_backward_reaches_model_parameters() -> None:
    torch.manual_seed(5)
    x = torch.randn(1, 5, 1, 2, 3, 3)
    times = torch.tensor([[0.0, 1.0, 2.2, 3.4, 5.0]])
    mask = torch.ones(1, 1, 2, 3, 3)
    model = Stage1TemporalWarpNet(num_basis=4, feature_channels=4)

    output = model(x, times, mask)
    output["x_warped"].sum().backward()

    final_weight = model.warp_head.layers[-1].weight
    assert final_weight.grad is not None
    assert torch.isfinite(final_weight.grad).all()
    assert final_weight.grad.abs().sum() > 0