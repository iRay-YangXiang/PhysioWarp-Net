"""Tests for non-aborting synthetic sweep verification."""

from __future__ import annotations

from physiowarp.training.debug_synthetic_stage1 import (
    SyntheticRunResult,
    validate_synthetic_result,
)


def test_validation_reports_multiple_failures() -> None:
    result = SyntheticRunResult(
        initial_total_loss=1.0,
        final_total_loss=1.1,
        initial_soft_success=0.5,
        final_soft_success=0.4,
        mean_abs_delta_inside=0.0,
        max_abs_delta_outside=0.2,
        final_image_loss=0.3,
        delta_saturation_fraction=0.5,
        temporal_clipping_fraction=0.4,
        monotonicity_violation_fraction=0.2,
    )
    verification = {
        "min_inside_delta_seconds": 0.01,
        "max_outside_delta_seconds": 1e-7,
        "max_image_l1": 0.1,
        "max_saturation_fraction": 0.25,
        "max_temporal_clipping_fraction": 0.15,
        "max_monotonicity_violation_fraction": 0.01,
    }

    failures = validate_synthetic_result(result, verification)

    assert len(failures) == 8
    assert "attack objective did not improve" in failures
    assert "total loss did not decrease" in failures


def test_validation_tolerates_float32_threshold_roundoff() -> None:
    result = SyntheticRunResult(
        initial_total_loss=1.0,
        final_total_loss=0.5,
        initial_soft_success=0.1,
        final_soft_success=0.9,
        mean_abs_delta_inside=0.1,
        max_abs_delta_outside=0.0,
        final_image_loss=0.05,
        delta_saturation_fraction=0.1,
        temporal_clipping_fraction=0.15000000596046448,
        monotonicity_violation_fraction=0.0,
    )
    verification = {
        "min_inside_delta_seconds": 0.01,
        "max_outside_delta_seconds": 1e-7,
        "max_image_l1": 0.1,
        "max_saturation_fraction": 0.25,
        "max_temporal_clipping_fraction": 0.15,
        "max_monotonicity_violation_fraction": 0.01,
    }

    assert validate_synthetic_result(result, verification) == []