"""Tests for temporal deformation diagnostics."""

from __future__ import annotations

import torch

from physiowarp.evaluation import deformation_diagnostics


def test_deformation_diagnostics_measure_known_failure_fractions() -> None:
    times = torch.tensor([0.0, 1.0, 2.0])
    delta = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 5.0],
        ]
    ).reshape(1, 3, 1, 1, 2)
    support = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2)

    metrics = deformation_diagnostics(
        times,
        delta,
        support,
        delta_max_seconds=1.0,
        saturation_tolerance=0.01,
        min_interval_seconds=0.1,
    )

    torch.testing.assert_close(
        metrics["mean_abs_delta_inside_seconds"], torch.tensor(2.0 / 3.0)
    )
    torch.testing.assert_close(metrics["max_abs_delta_outside_seconds"], torch.tensor(5.0))
    torch.testing.assert_close(metrics["delta_saturation_fraction"], torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(metrics["temporal_clipping_fraction"], torch.tensor(0.0))
    torch.testing.assert_close(
        metrics["monotonicity_violation_fraction"], torch.tensor(0.5)
    )


def test_deformation_diagnostics_detect_temporal_clipping() -> None:
    times = torch.tensor([0.0, 1.0, 2.0])
    delta = torch.tensor([-0.2, 0.0, 0.2]).reshape(1, 3, 1, 1, 1)
    support = torch.ones(1, 1, 1, 1, 1)

    metrics = deformation_diagnostics(
        times,
        delta,
        support,
        delta_max_seconds=1.0,
    )

    torch.testing.assert_close(
        metrics["temporal_clipping_fraction"], torch.tensor(2.0 / 3.0)
    )