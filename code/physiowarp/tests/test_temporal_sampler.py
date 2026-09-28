"""Tests for differentiable nonuniform temporal interpolation."""

from __future__ import annotations

import torch

from physiowarp.warping.temporal_sampler import temporal_warp


def _signal(values: list[float]) -> torch.Tensor:
    return torch.tensor(values).reshape(1, len(values), 1, 1, 1, 1)


def test_zero_delta_is_identity() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 4, 2, 2, 3, 3)
    times = torch.tensor([[0.0, 1.0, 2.5, 5.0], [0.0, 0.7, 2.0, 4.0]])
    delta = torch.zeros(2, 4, 2, 3, 3)

    warped = temporal_warp(x, times, delta)

    torch.testing.assert_close(warped, x)


def test_positive_delta_samples_later_on_linear_curve() -> None:
    x = _signal([0.0, 10.0, 20.0, 30.0])
    times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    delta = torch.full((1, 4, 1, 1, 1), 0.5)

    warped = temporal_warp(x, times, delta)

    expected = _signal([5.0, 15.0, 25.0, 30.0])
    torch.testing.assert_close(warped, expected)


def test_nonuniform_acquisition_times() -> None:
    times = torch.tensor([0.0, 1.0, 2.5, 5.0])
    x = _signal([0.0, 10.0, 25.0, 50.0])
    delta = torch.full((1, 4, 1, 1, 1), 0.5)

    warped = temporal_warp(x, times, delta)

    expected = _signal([5.0, 15.0, 30.0, 50.0])
    torch.testing.assert_close(warped, expected)


def test_queries_are_clipped_to_temporal_boundaries() -> None:
    x = _signal([0.0, 10.0, 20.0, 30.0])
    times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    delta = torch.tensor([-10.0, 10.0, -10.0, 10.0]).reshape(1, 4, 1, 1, 1)

    warped = temporal_warp(x, times, delta)

    expected = _signal([0.0, 30.0, 0.0, 30.0])
    torch.testing.assert_close(warped, expected)


def test_gradient_with_respect_to_delta_is_finite_and_nonzero() -> None:
    x = _signal([0.0, 10.0, 20.0, 30.0])
    times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    delta = torch.full((1, 4, 1, 1, 1), 0.25, requires_grad=True)

    temporal_warp(x, times, delta).sum().backward()

    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_delta_with_singleton_channel_dimension_is_supported() -> None:
    x = _signal([0.0, 10.0, 20.0, 30.0])
    times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    delta = torch.zeros(1, 4, 1, 1, 1, 1)

    torch.testing.assert_close(temporal_warp(x, times, delta), x)