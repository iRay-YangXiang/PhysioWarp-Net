"""Tests for temporal deformation losses."""

from __future__ import annotations

import torch

from physiowarp.losses.temporal_loss import (
    first_order_delta_loss,
    mean_abs_delta,
    second_order_delta_loss,
)


def test_temporal_differences_have_expected_values() -> None:
    delta = torch.tensor([0.0, 1.0, 4.0]).reshape(1, 3, 1, 1, 1)

    torch.testing.assert_close(first_order_delta_loss(delta), torch.tensor(2.0))
    torch.testing.assert_close(second_order_delta_loss(delta), torch.tensor(2.0))


def test_mask_excludes_values_outside_roi() -> None:
    delta = torch.tensor([[[[[1.0, 100.0]]], [[[3.0, 100.0]]], [[[5.0, 100.0]]]]])
    mask = torch.tensor([[[[[1.0, 0.0]]]]])

    torch.testing.assert_close(mean_abs_delta(delta, mask), torch.tensor(3.0))
    torch.testing.assert_close(first_order_delta_loss(delta, mask), torch.tensor(2.0))
    torch.testing.assert_close(second_order_delta_loss(delta, mask), torch.tensor(0.0))