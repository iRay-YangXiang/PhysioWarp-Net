"""Tests for smooth quantitative threshold attacks."""

from __future__ import annotations

import pytest
import torch

from physiowarp.losses.attack_loss import AttackLossConfig, ThresholdAttackLoss


@pytest.mark.parametrize(
    ("mode", "baseline", "successful", "unsuccessful"),
    [
        ("rcbf_down", 0.5, 0.2, 0.5),
        ("rcbf_up", 0.1, 0.4, 0.1),
        ("tmax_up", 4.0, 7.0, 4.0),
        ("tmax_down", 8.0, 5.0, 8.0),
    ],
)
def test_all_modes_reward_threshold_crossing(
    mode: str,
    baseline: float,
    successful: float,
    unsuccessful: float,
) -> None:
    baseline_map = torch.full((1, 1, 1, 1), baseline)
    masks = torch.ones_like(baseline_map)
    objective = ThresholdAttackLoss(AttackLossConfig(mode=mode))  # type: ignore[arg-type]

    success = objective(
        baseline_map,
        torch.full_like(baseline_map, successful),
        masks,
        masks,
    )
    failure = objective(
        baseline_map,
        torch.full_like(baseline_map, unsuccessful),
        masks,
        masks,
    )

    assert success["loss"] < failure["loss"]
    torch.testing.assert_close(success["hard_success_fraction"], torch.tensor(1.0))


def test_baseline_margin_excludes_borderline_voxels() -> None:
    baseline = torch.tensor([0.2, 5.8]).reshape(1, 1, 1, 2)
    attacked = torch.tensor([7.0, 7.0]).reshape(1, 1, 1, 2)
    mask = torch.ones_like(baseline)
    objective = ThresholdAttackLoss(AttackLossConfig(mode="tmax_up"))

    result = objective(baseline, attacked, mask, mask)

    torch.testing.assert_close(result["num_candidate_voxels"], torch.tensor(1.0))


def test_attack_loss_gradient_is_finite_and_points_toward_success() -> None:
    baseline = torch.full((1, 1, 1, 1), 4.0)
    attacked = torch.full((1, 1, 1, 1), 5.0, requires_grad=True)
    mask = torch.ones_like(baseline)
    objective = ThresholdAttackLoss(AttackLossConfig(mode="tmax_up"))

    objective(baseline, attacked, mask, mask)["loss"].backward()

    assert attacked.grad is not None
    assert torch.isfinite(attacked.grad).all()
    assert attacked.grad.item() < 0


def test_no_candidates_returns_differentiable_zero_loss() -> None:
    baseline = torch.full((1, 1, 1, 1), 7.0)
    attacked = torch.full((1, 1, 1, 1), 7.0, requires_grad=True)
    mask = torch.ones_like(baseline)
    objective = ThresholdAttackLoss(AttackLossConfig(mode="tmax_up"))

    result = objective(baseline, attacked, mask, mask)
    result["loss"].backward()

    torch.testing.assert_close(result["loss"], torch.tensor(0.0))
    assert attacked.grad is not None
