"""Run and summarize the synthetic Stage 1 experiment across random seeds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from .debug_synthetic_stage1 import (
    _merge_config,
    run_synthetic_experiment,
    validate_synthetic_result,
)


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream)
    return config


def _evaluation_settings(
    config: dict[str, Any],
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    seeds = [int(seed) for seed in config["evaluation"]["seeds"]]
    if len(seeds) < 2:
        raise ValueError("evaluation.seeds must contain at least two seeds")
    scenarios = config["evaluation"]["scenarios"]
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("evaluation.scenarios must be a non-empty mapping")
    return seeds, scenarios


def main() -> None:
    """Execute all configured seeds and write machine-readable results."""
    default_config = Path(__file__).parents[1] / "configs" / "stage1_synthetic.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "stage1_robustness",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)
    seeds, scenarios = _evaluation_settings(config)
    run_records: list[dict[str, Any]] = []
    for scenario_name, overrides in scenarios.items():
        for seed in seeds:
            result = run_synthetic_experiment(
                args.config,
                seed=seed,
                output_dir=args.output_dir / scenario_name / f"seed_{seed}",
                config_overrides=overrides,
                verify=False,
            )
            record: dict[str, Any] = asdict(result)
            record["scenario"] = scenario_name
            record["seed"] = seed
            scenario_config = _merge_config(config, overrides)
            failures = validate_synthetic_result(
                result,
                scenario_config["verification"],
            )
            record["passed"] = not failures
            record["failures"] = failures
            run_records.append(record)

    metric_names = list(asdict(result))
    summary = {
        scenario_name: {
            name: {
                "mean": mean(
                    record[name]
                    for record in run_records
                    if record["scenario"] == scenario_name
                ),
                "std": stdev(
                    record[name]
                    for record in run_records
                    if record["scenario"] == scenario_name
                ),
            }
            for name in metric_names
        }
        for scenario_name in scenarios
    }
    payload = {
        "all_passed": all(record["passed"] for record in run_records),
        "runs": run_records,
        "summary": summary,
    }
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
