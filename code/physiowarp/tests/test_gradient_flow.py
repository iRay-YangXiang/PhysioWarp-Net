"""Finite-difference checks for temporal deformation gradients."""

from __future__ import annotations

import torch

from physiowarp.warping.temporal_sampler import temporal_warp


def test_sampler_autograd_matches_finite_difference() -> None:
    dtype = torch.float64
    x = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=dtype).reshape(
        1, 4, 1, 1, 1, 1
    )
    times = torch.tensor([0.0, 1.0, 2.5, 4.0], dtype=dtype)
    delta = torch.full((1, 4, 1, 1, 1), 0.2, dtype=dtype, requires_grad=True)
    target_index = (0, 1, 0, 0, 0)

    loss = temporal_warp(x, times, delta).square().sum()
    loss.backward()
    assert delta.grad is not None
    autograd_value = delta.grad[target_index]

    epsilon = 1e-6
    with torch.no_grad():
        delta_plus = delta.detach().clone()
        delta_minus = delta.detach().clone()
        delta_plus[target_index] += epsilon
        delta_minus[target_index] -= epsilon
        loss_plus = temporal_warp(x, times, delta_plus).square().sum()
        loss_minus = temporal_warp(x, times, delta_minus).square().sum()
    finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon)

    relative_error = torch.abs(autograd_value - finite_difference) / torch.maximum(
        torch.abs(finite_difference),
        torch.tensor(1e-12, dtype=dtype),
    )
    assert relative_error < 1e-6