"""Tests for smooth temporal basis evaluation."""

from __future__ import annotations

import pytest
import torch

from physiowarp.warping.temporal_basis import TemporalBSplineBasis


def test_basis_shape_partition_and_endpoints() -> None:
    times = torch.tensor([[0.0, 0.4, 1.7, 5.0], [2.0, 3.0, 4.5, 9.0]])
    basis = TemporalBSplineBasis(num_basis=5, degree=3)(times)

    assert basis.shape == (2, 4, 5)
    torch.testing.assert_close(basis.sum(dim=-1), torch.ones(2, 4))
    torch.testing.assert_close(basis[:, 0, 0], torch.ones(2))
    torch.testing.assert_close(basis[:, -1, -1], torch.ones(2))


def test_basis_supports_unbatched_times_and_gradients() -> None:
    times = torch.tensor([0.0, 0.8, 2.5, 5.0], requires_grad=True)
    basis = TemporalBSplineBasis(num_basis=4)(times)
    loss = (basis * torch.arange(4, dtype=times.dtype)).sum()
    loss.backward()

    assert basis.shape == (1, 4, 4)
    assert times.grad is not None
    assert torch.isfinite(times.grad).all()


def test_basis_rejects_non_increasing_times() -> None:
    times = torch.tensor([0.0, 1.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        TemporalBSplineBasis(num_basis=4)(times)