"""Train Stage 1 end-to-end without medical data."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from physiowarp.evaluation import deformation_diagnostics
from physiowarp.evaluation.visualize import save_stage1_summary
from physiowarp.losses.attack_loss import AttackLossConfig, ThresholdAttackLoss
from physiowarp.losses.total_loss import LossWeights, stage1_total_loss
from physiowarp.models import Stage1TemporalWarpNet
from physiowarp.perfusion import SoftTemporalSummaryPerfusion
from physiowarp.utils import set_seed


@dataclass(frozen=True)
class SyntheticRunResult:
    """Summary of the synthetic optimization checks."""

    initial_total_loss: float
    final_total_loss: float
    initial_soft_success: float
    final_soft_success: float
    mean_abs_delta_inside: float
    max_abs_delta_outside: float
    final_image_loss: float
    delta_saturation_fraction: float
    temporal_clipping_fraction: float
    monotonicity_violation_fraction: float


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _merge_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    if overrides is None:
        return merged
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def _make_synthetic_batch(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create smooth Gaussian TACs and a central fixed ROI."""
    batch_size = int(config["batch_size"])
    num_times = int(config["num_times"])
    depth = int(config["depth"])
    height = int(config["height"])
    width = int(config["width"])
    end_time = float(config["end_time_seconds"])
    peak_time = float(config["peak_time_seconds"])
    peak_width = float(config["peak_width_seconds"])

    jitter_fraction = float(config.get("time_jitter_fraction", 0.0))
    if not 0.0 <= jitter_fraction < 1.0:
        raise ValueError("time_jitter_fraction must lie in [0, 1)")
    interval_indices = torch.arange(num_times - 1, device=device, dtype=torch.float32)
    intervals = 1.0 + jitter_fraction * torch.sin(1.7 * interval_indices + 0.3)
    times = torch.cat((torch.zeros(1, device=device), intervals.cumsum(dim=0)))
    times = times * (end_time / times[-1])
    z_grid = torch.linspace(-1.0, 1.0, depth, device=device)
    y_grid = torch.linspace(-1.0, 1.0, height, device=device)
    x_grid = torch.linspace(-1.0, 1.0, width, device=device)
    zz, yy, xx = torch.meshgrid(z_grid, y_grid, x_grid, indexing="ij")
    amplitude = 0.8 + 0.2 * torch.exp(-(zz.square() + yy.square() + xx.square()))
    curve = torch.exp(-0.5 * ((times - peak_time) / peak_width).square())
    x = curve[None, :, None, None, None, None] * amplitude[None, None, None]
    x = x.expand(batch_size, -1, -1, -1, -1, -1).contiguous()

    roi_fraction = float(config["roi_fraction"])
    roi_height = max(1, round(height * roi_fraction))
    roi_width = max(1, round(width * roi_fraction))
    height_start = (height - roi_height) // 2
    width_start = (width - roi_width) // 2
    mask = torch.zeros(batch_size, 1, depth, height, width, device=device)
    mask[
        :,
        :,
        :,
        height_start : height_start + roi_height,
        width_start : width_start + roi_width,
    ] = 1.0
    return x, times, mask


def run_synthetic_experiment(
    config_path: Path,
    *,
    seed: int | None = None,
    output_dir: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    verify: bool = True,
) -> SyntheticRunResult:
    """Optimize and verify a fully synthetic Stage 1 experiment."""
    config = _merge_config(_load_config(config_path), config_overrides)
    training_config = config["training"]
    requested_device = str(training_config["device"])
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    set_seed(
        int(training_config["seed"] if seed is None else seed),
        deterministic=bool(training_config["deterministic"]),
    )

    x, times, fixed_mask = _make_synthetic_batch(config["synthetic"], device)
    model_config = config["model"]
    model = Stage1TemporalWarpNet(
        num_basis=int(model_config["num_basis"]),
        delta_max_seconds=float(model_config["delta_max_seconds"]),
        feature_channels=int(model_config["feature_channels"]),
        spline_degree=int(model_config["spline_degree"]),
    ).to(device)
    victim = SoftTemporalSummaryPerfusion(
        temperature=float(config["surrogate"]["temperature"])
    ).to(device)
    attack_config = AttackLossConfig(**config["attack"])
    attack_objective = ThresholdAttackLoss(attack_config)
    loss_weights = LossWeights(**config["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["lr"]))

    with torch.no_grad():
        baseline_maps = victim(x, times)
    target_name = "tmax" if attack_config.mode.startswith("tmax") else "rcbf"
    candidate_mask = torch.ones_like(baseline_maps[target_name])
    support_mask = fixed_mask[:, 0]

    initial_total = 0.0
    initial_success = 0.0
    final_losses: dict[str, torch.Tensor] = {}
    final_attack: dict[str, torch.Tensor] = {}
    final_output: dict[str, torch.Tensor] = {}
    num_iterations = int(training_config["iterations"])
    log_every = int(training_config["log_every"])
    for iteration in range(num_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, times, fixed_mask)
        attacked_maps = victim(output["x_warped"], times)
        attack_metrics = attack_objective(
            baseline_maps[target_name],
            attacked_maps[target_name],
            candidate_mask,
            support_mask,
        )
        losses = stage1_total_loss(
            x,
            output["x_warped"],
            times,
            output["delta_effective"],
            attack_metrics["loss"],
            fixed_mask=fixed_mask,
            weights=loss_weights,
            min_interval_seconds=float(training_config["min_interval_seconds"]),
        )
        if iteration == 0:
            initial_total = losses["total_loss"].item()
            initial_success = attack_metrics["soft_success"].item()
        if iteration == num_iterations:
            final_losses = losses
            final_attack = attack_metrics
            final_output = output
            break

        losses["total_loss"].backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise RuntimeError("non-finite model gradient encountered")
        optimizer.step()

        if iteration % log_every == 0:
            print(
                f"iteration={iteration:04d} "
                f"total_loss={losses['total_loss'].item():.6f} "
                f"attack_loss={losses['attack_loss'].item():.6f} "
                f"image_loss={losses['image_loss'].item():.6f} "
                f"delta_first_order={losses['delta_first_order'].item():.6f} "
                f"delta_second_order={losses['delta_second_order'].item():.6f} "
                f"monotonicity_loss={losses['monotonicity_loss'].item():.6f} "
                f"mean_abs_delta_seconds={losses['mean_abs_delta_seconds'].item():.6f} "
                f"max_abs_delta_seconds={losses['max_abs_delta_seconds'].item():.6f} "
                f"threshold_soft_success={attack_metrics['soft_success'].item():.6f} "
                f"threshold_hard_success={attack_metrics['hard_success_fraction'].item():.6f} "
                f"candidate_voxel_count={attack_metrics['num_candidate_voxels'].item():.0f}"
            )

    diagnostics = deformation_diagnostics(
        times,
        final_output["delta_effective"],
        fixed_mask,
        delta_max_seconds=float(model_config["delta_max_seconds"]),
        saturation_tolerance=float(config["evaluation"]["saturation_tolerance"]),
        min_interval_seconds=float(training_config["min_interval_seconds"]),
    )
    result = SyntheticRunResult(
        initial_total_loss=initial_total,
        final_total_loss=final_losses["total_loss"].item(),
        initial_soft_success=initial_success,
        final_soft_success=final_attack["soft_success"].item(),
        mean_abs_delta_inside=diagnostics["mean_abs_delta_inside_seconds"].item(),
        max_abs_delta_outside=diagnostics["max_abs_delta_outside_seconds"].item(),
        final_image_loss=final_losses["image_loss"].item(),
        delta_saturation_fraction=diagnostics["delta_saturation_fraction"].item(),
        temporal_clipping_fraction=diagnostics["temporal_clipping_fraction"].item(),
        monotonicity_violation_fraction=diagnostics[
            "monotonicity_violation_fraction"
        ].item(),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_stage1_summary(
            x,
            final_output["x_warped"],
            times,
            fixed_mask,
            final_output["delta_effective"],
            baseline_maps[target_name],
            attacked_maps[target_name],
            output_dir / "stage1_summary.png",
        )

    failures = validate_synthetic_result(result, config["verification"])
    if verify and failures:
        raise RuntimeError("; ".join(failures))
    return result


def validate_synthetic_result(
    result: SyntheticRunResult,
    verification: dict[str, Any],
) -> list[str]:
    """Return configured verification failures without stopping a sweep."""
    tolerance = 1e-6
    failures: list[str] = []
    if result.final_soft_success <= result.initial_soft_success + tolerance:
        failures.append("attack objective did not improve")
    if result.mean_abs_delta_inside < (
        float(verification["min_inside_delta_seconds"]) - tolerance
    ):
        failures.append("learned displacement remained too small inside ROI")
    if result.max_abs_delta_outside > (
        float(verification["max_outside_delta_seconds"]) + tolerance
    ):
        failures.append("displacement leaked outside fixed ROI")
    if result.final_image_loss > float(verification["max_image_l1"]) + tolerance:
        failures.append("image difference exceeded configured limit")
    if result.delta_saturation_fraction > float(
        verification["max_saturation_fraction"]
    ) + tolerance:
        failures.append("too much displacement saturated at the configured bound")
    if result.temporal_clipping_fraction > float(
        verification["max_temporal_clipping_fraction"]
    ) + tolerance:
        failures.append("too many temporal queries required boundary clipping")
    if result.monotonicity_violation_fraction > float(
        verification["max_monotonicity_violation_fraction"]
    ) + tolerance:
        failures.append("transformed time violated monotonicity too often")
    if result.final_total_loss >= result.initial_total_loss - tolerance:
        failures.append("total loss did not decrease")
    return failures


def main() -> None:
    """Run the configured synthetic experiment and print its final summary."""
    default_config = Path(__file__).parents[1] / "configs" / "stage1_synthetic.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_synthetic_experiment(
        args.config,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
