"""Tests for temporal monotonicity regularization."""

from __future__ import annotations

import pytest
import torch

from physiowarp.warping.monotonicity import temporal_monotonicity_loss


def test_unmodified_times_have_no_violations() -> None:
    times = torch.tensor([[0.0, 0.7, 2.0, 4.5]])
    delta = torch.zeros(1, 4, 2, 2, 2)

    loss, fraction = temporal_monotonicity_loss(
        times,
        delta,
        min_interval_seconds=0.1,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(fraction, torch.tensor(0.0))


def test_reversed_transformed_time_has_expected_penalty() -> None:
    times = torch.tensor([0.0, 1.0, 2.0])
    delta = torch.tensor([0.0, 0.0, -1.5]).reshape(1, 3, 1, 1, 1)

    loss, fraction = temporal_monotonicity_loss(
        times,
        delta,
        min_interval_seconds=0.1,
    )

    torch.testing.assert_close(loss, torch.tensor(0.3))
    torch.testing.assert_close(fraction, torch.tensor(0.5))


def test_monotonicity_loss_has_finite_nonzero_gradient() -> None:
    times = torch.tensor([0.0, 1.0, 2.0])
    delta = torch.tensor([0.0, 0.0, -1.5]).reshape(1, 3, 1, 1, 1)
    delta.requires_grad_(True)

    loss, _ = temporal_monotonicity_loss(times, delta)
    loss.backward()

    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_negative_minimum_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        temporal_monotonicity_loss(
            torch.tensor([0.0, 1.0]),
            torch.zeros(1, 2, 1, 1, 1),
            min_interval_seconds=-0.1,
        )