"""Smooth temporal basis functions for displacement fields."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBSplineBasis(nn.Module):
    """Evaluate a clamped B-spline basis at batched acquisition times.

    The knot vector is fixed and uniform in normalized time. Physical times
    are normalized independently for each batch item, which permits irregular
    and case-specific acquisition schedules.

    Args:
        num_basis: Number of basis functions ``K``.
        degree: Polynomial degree. Cubic splines use degree 3.
        eps: Minimum accepted physical time range.

    Input:
        acquisition_times: Strictly increasing times with shape ``[B, T]`` or
            ``[T]``.

    Output:
        Basis matrix with shape ``[B, T, K]``.
    """

    def __init__(
        self,
        num_basis: int,
        degree: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_basis < 2:
            raise ValueError("num_basis must be at least 2")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if num_basis <= degree:
            raise ValueError("num_basis must be greater than degree")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_basis = num_basis
        self.degree = degree
        self.eps = eps
        self.register_buffer("knots", self._make_knots(num_basis, degree))

    @staticmethod
    def _make_knots(num_basis: int, degree: int) -> torch.Tensor:
        num_interior = num_basis - degree - 1
        interior = torch.linspace(0.0, 1.0, num_interior + 2)[1:-1]
        return torch.cat(
            (
                torch.zeros(degree + 1),
                interior,
                torch.ones(degree + 1),
            )
        )

    def forward(self, acquisition_times: torch.Tensor) -> torch.Tensor:
        """Return basis values ``[B, T, K]`` for times ``[B, T]`` or ``[T]``."""
        if acquisition_times.ndim == 1:
            acquisition_times = acquisition_times.unsqueeze(0)
        if acquisition_times.ndim != 2:
            raise ValueError("acquisition_times must have shape [T] or [B, T]")
        if acquisition_times.shape[1] < 2:
            raise ValueError("at least two acquisition times are required")
        if not acquisition_times.is_floating_point():
            raise TypeError("acquisition_times must be floating point")
        if not torch.isfinite(acquisition_times).all():
            raise ValueError("acquisition_times must contain only finite values")

        intervals = acquisition_times[:, 1:] - acquisition_times[:, :-1]
        if torch.any(intervals <= 0):
            raise ValueError("acquisition_times must be strictly increasing")

        start = acquisition_times[:, :1]
        duration = acquisition_times[:, -1:] - start
        if torch.any(duration <= self.eps):
            raise ValueError("acquisition time range is too small")
        normalized_times = (acquisition_times - start) / duration

        knots = self.knots.to(
            device=acquisition_times.device,
            dtype=acquisition_times.dtype,
        )
        values = normalized_times.unsqueeze(-1)
        basis = (
            (values >= knots[:-1]) & (values < knots[1:])
        ).to(acquisition_times.dtype)
        basis[..., self.num_basis - 1] = torch.where(
            normalized_times == 1,
            torch.ones_like(normalized_times),
            basis[..., self.num_basis - 1],
        )

        for current_degree in range(1, self.degree + 1):
            next_basis = torch.zeros(
                (*normalized_times.shape, basis.shape[-1] - 1),
                device=acquisition_times.device,
                dtype=acquisition_times.dtype,
            )
            for basis_index in range(next_basis.shape[-1]):
                left_denominator = (
                    knots[basis_index + current_degree] - knots[basis_index]
                )
                right_denominator = (
                    knots[basis_index + current_degree + 1]
                    - knots[basis_index + 1]
                )
                if left_denominator > 0:
                    next_basis[..., basis_index] += (
                        (normalized_times - knots[basis_index])
                        / left_denominator
                        * basis[..., basis_index]
                    )
                if right_denominator > 0:
                    next_basis[..., basis_index] += (
                        (knots[basis_index + current_degree + 1] - normalized_times)
                        / right_denominator
                        * basis[..., basis_index + 1]
                    )
            basis = next_basis

        basis[..., -1] = torch.where(
            normalized_times == 1,
            torch.ones_like(normalized_times),
            basis[..., -1],
        )
        return basis
