"""Matplotlib visualizations for temporal deformation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def save_stage1_summary(
    original: torch.Tensor,
    warped: torch.Tensor,
    acquisition_times: torch.Tensor,
    fixed_mask: torch.Tensor,
    delta: torch.Tensor,
    baseline_map: torch.Tensor,
    attacked_map: torch.Tensor,
    output_path: Path,
) -> None:
    """Save spatial maps and representative curves for one Stage 1 case.

    Images have shape ``[B,T,1,Z,H,W]``; mask is ``[B,1,Z,H,W]``;
    displacement is ``[B,T,Z,H,W]``; quantitative maps are ``[B,Z,H,W]``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_index = 0
    mask = fixed_mask[batch_index, 0]
    depth_index = mask.sum(dim=(-1, -2)).argmax().item()
    spatial_mask = mask[depth_index]
    nonzero = torch.nonzero(spatial_mask > 0, as_tuple=False)
    if nonzero.numel() == 0:
        raise ValueError("fixed_mask must contain at least one supported voxel")
    representative = nonzero[len(nonzero) // 2]
    row, column = representative.tolist()
    times = acquisition_times
    if times.ndim == 2:
        times = times[batch_index]
    frame_index = original[batch_index, :, 0, depth_index].mean(dim=(-1, -2)).argmax().item()

    original_frame = original[batch_index, frame_index, 0, depth_index]
    warped_frame = warped[batch_index, frame_index, 0, depth_index]
    difference = (warped_frame - original_frame).abs()
    mean_delta = delta[batch_index, :, depth_index].mean(dim=0)
    max_delta = delta[batch_index, :, depth_index].abs().amax(dim=0)
    original_curve = original[batch_index, :, 0, depth_index, row, column]
    warped_curve = warped[batch_index, :, 0, depth_index, row, column]
    delta_curve = delta[batch_index, :, depth_index, row, column]

    arrays = [
        original_frame,
        warped_frame,
        difference,
        spatial_mask,
        mean_delta,
        max_delta,
        baseline_map[batch_index, depth_index],
        attacked_map[batch_index, depth_index],
    ]
    titles = [
        "Original frame",
        "Warped frame",
        "Absolute difference",
        "Fixed ROI",
        "Mean delta (s)",
        "Max |delta| (s)",
        "Baseline target",
        "Attacked target",
    ]
    image_min = min(original_frame.min().item(), warped_frame.min().item())
    image_max = max(original_frame.max().item(), warped_frame.max().item())
    map_min = min(baseline_map.min().item(), attacked_map.min().item())
    map_max = max(baseline_map.max().item(), attacked_map.max().item())
    color_limits: list[tuple[float | None, float | None]] = [
        (image_min, image_max),
        (image_min, image_max),
        (0.0, difference.max().item()),
        (0.0, 1.0),
        (None, None),
        (0.0, delta.abs().max().item()),
        (map_min, map_max),
        (map_min, map_max),
    ]
    figure, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for axis, array, title, limits in zip(
        axes.flat[:8], arrays, titles, color_limits, strict=True
    ):
        image = axis.imshow(
            array.detach().cpu().numpy(),
            cmap="viridis",
            vmin=limits[0],
            vmax=limits[1],
        )
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046)

    time_values = times.detach().cpu().numpy()
    axes[2, 0].plot(time_values, original_curve.detach().cpu().numpy(), label="original")
    axes[2, 0].plot(time_values, warped_curve.detach().cpu().numpy(), label="warped")
    axes[2, 0].set_title("Representative TAC")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].legend()
    axes[2, 1].plot(time_values, delta_curve.detach().cpu().numpy())
    axes[2, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[2, 1].set_title("Representative delta(t)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Seconds")
    axes[2, 2].axis("off")
    axes[2, 3].axis("off")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
