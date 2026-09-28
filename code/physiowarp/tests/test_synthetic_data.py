"""Tests for synthetic experiment data generation."""

from __future__ import annotations

import torch

from physiowarp.training.debug_synthetic_stage1 import _make_synthetic_batch


def test_irregular_synthetic_times_are_strict_and_preserve_range() -> None:
    config = {
        "batch_size": 1,
        "num_times": 10,
        "depth": 2,
        "height": 8,
        "width": 8,
        "end_time_seconds": 9.0,
        "peak_time_seconds": 5.0,
        "peak_width_seconds": 1.0,
        "roi_fraction": 0.5,
        "time_jitter_fraction": 0.35,
    }

    x, times, mask = _make_synthetic_batch(config, torch.device("cpu"))
    intervals = times[1:] - times[:-1]

    assert x.shape == (1, 10, 1, 2, 8, 8)
    assert mask.shape == (1, 1, 2, 8, 8)
    assert torch.all(intervals > 0)
    assert not torch.allclose(intervals, intervals.mean().expand_as(intervals))
    torch.testing.assert_close(times[[0, -1]], torch.tensor([0.0, 9.0]))