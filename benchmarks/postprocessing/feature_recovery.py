"""Aggregate and plot the feature-recovery benchmark.

The command accepts an incomplete result directory, which makes it useful for
previewing the final figure design before all production seeds have finished.
Smoke runs, failed runs, and malformed outputs are excluded automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tomllib

from benchmarks.postprocessing.style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    publication_style,
)

plt.switch_backend("Agg")


@dataclass(frozen=True)
class FeatureRecoveryRecord:
    """One selected production model in tidy, analysis-ready form."""

    source_file: str
    job_id: str
    method: str
    seed: int
    g1: float
    g2: float
    g1_absolute_error: float
    g2_absolute_error: float
    g1_relative_error: float
    g2_relative_error: float
    joint_exponent_error: float
    validation_nrmse: float
    test_nrmse: float
    coefficient_relative_l2_error: float
    support_accuracy: float
    active_coefficients: int
    candidate_evaluations: int
    failed_candidates: int
    failed_candidate_fraction: float
    wall_time_seconds: float
    git_commit: str
    git_dirty: bool


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "complete" or payload.get("mode") != "production":
        return None
    test_nrmse = payload.get("test", {}).get("rollout_nrmse")
    if not _finite_number(test_nrmse):
        return None
    return payload


def _record_from_payload(
    payload: dict[str, Any],
    source: Path,
    config: dict[str, Any],
) -> FeatureRecoveryRecord:
    job = payload["job"]
    selected = payload["selected"]
    variables = selected["variables"]
    method = job["method"]
    if method == "fixed_library":
        g1 = float(config["fixed_library"]["g1"])
        g2 = float(config["fixed_library"]["g2"])
    else:
        g1 = float(variables["g1"])
        g2 = float(variables["g2"])

    truth_g1 = float(config["truth"]["g1"])
    truth_g2 = float(config["truth"]["g2"])
    g1_error = abs(g1 - truth_g1)
    g2_error = abs(g2 - truth_g2)
    evaluations = int(payload["actual_candidate_evaluations"])
    failed = int(payload["failed_candidate_count"])
    coefficient_metrics = selected["coefficient_metrics"]
    provenance = payload["provenance"]

    return FeatureRecoveryRecord(
        source_file=str(source),
        job_id=str(job["job_id"]),
        method=method,
        seed=int(job["data_seed"]),
        g1=g1,
        g2=g2,
        g1_absolute_error=g1_error,
        g2_absolute_error=g2_error,
        g1_relative_error=g1_error / abs(truth_g1),
        g2_relative_error=g2_error / abs(truth_g2),
        joint_exponent_error=float(np.hypot(g1_error, g2_error)),
        validation_nrmse=float(selected["validation"]["rollout_nrmse"]),
        test_nrmse=float(payload["test"]["rollout_nrmse"]),
        coefficient_relative_l2_error=float(coefficient_metrics["relative_l2_error"]),
        support_accuracy=float(coefficient_metrics["support_accuracy"]),
        active_coefficients=int(coefficient_metrics["active_coefficients"]),
        candidate_evaluations=evaluations,
        failed_candidates=failed,
        failed_candidate_fraction=failed / evaluations if evaluations else math.nan,
        wall_time_seconds=float(payload["wall_time_seconds"]),
        git_commit=str(provenance["git_commit"]),
        git_dirty=bool(provenance["git_dirty"]),
    )


def collect_records(
    raw_directory: Path,
    config: dict[str, Any],
) -> tuple[list[FeatureRecoveryRecord], list[dict[str, Any]]]:
    """Collect valid production records and their original payloads."""
    records = []
    payloads = []
    for path in sorted(raw_directory.glob("*.json")):
        payload = _read_payload(path)
        if payload is None:
            continue
        if payload.get("job", {}).get("experiment") != "feature_recovery":
            continue
        records.append(_record_from_payload(payload, path, config))
        payloads.append(payload)
    return records, payloads


def write_summary_csv(path: Path, records: list[FeatureRecoveryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _campaign_label(records: list[FeatureRecoveryRecord], expected_seeds: int) -> str:
    observed = len({record.seed for record in records})
    dirty = any(record.git_dirty for record in records)
    state = "PREVIEW" if observed < expected_seeds or dirty else "FINAL"
    return f"{state} — {observed}/{expected_seeds} seeds"


def _offsets(count: int, width: float = 0.12) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def _method_records(
    records: list[FeatureRecoveryRecord], method: str
) -> list[FeatureRecoveryRecord]:
    return sorted(
        (record for record in records if record.method == method),
        key=lambda record: record.seed,
    )


def plot_summary(
    output_stem: Path,
    records: list[FeatureRecoveryRecord],
    config: dict[str, Any],
) -> None:
    """Create the main parameter-recovery and predictive-accuracy figure."""
    expected_seeds = len(config["experiment"]["seeds"])
    methods = [
        method
        for method in config["experiment"]["methods"]
        if _method_records(records, method)
    ]
    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
        parameter_axis, error_axis = axes

        parameter_positions = {"g1": 0.0, "g2": 1.0}
        truth = {
            "g1": float(config["truth"]["g1"]),
            "g2": float(config["truth"]["g2"]),
        }
        method_shifts = np.linspace(-0.10, 0.10, max(len(methods), 1))
        for shift, method in zip(method_shifts, methods):
            method_records = _method_records(records, method)
            for parameter in ("g1", "g2"):
                values = np.asarray(
                    [getattr(record, parameter) for record in method_records]
                )
                x_values = (
                    parameter_positions[parameter]
                    + shift
                    + _offsets(len(values), width=0.035)
                )
                parameter_axis.scatter(
                    x_values,
                    values,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    facecolors="none"
                    if method == "fixed_library"
                    else METHOD_COLORS[method],
                    linewidths=1.1,
                    label=METHOD_LABELS[method] if parameter == "g1" else None,
                    zorder=3,
                )
        for parameter, position in parameter_positions.items():
            parameter_axis.hlines(
                truth[parameter],
                position - 0.30,
                position + 0.30,
                color="#222222",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
        parameter_axis.set_xticks([0, 1], [r"$g_1$", r"$g_2$"])
        parameter_axis.set_ylabel("Recovered exponent")
        parameter_axis.set_title("a  Parameter recovery", loc="left", fontweight="bold")
        parameter_axis.legend(loc="best")
        parameter_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)

        metric_positions = {"Validation": 0.0, "Test": 1.0}
        for shift, method in zip(method_shifts, methods):
            method_records = _method_records(records, method)
            for metric, position in metric_positions.items():
                attribute = (
                    "validation_nrmse" if metric == "Validation" else "test_nrmse"
                )
                values = np.asarray(
                    [getattr(record, attribute) for record in method_records]
                )
                x_values = position + shift + _offsets(len(values), width=0.035)
                error_axis.scatter(
                    x_values,
                    values,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    facecolors="none"
                    if method == "fixed_library"
                    else METHOD_COLORS[method],
                    linewidths=1.1,
                    zorder=3,
                )
                if len(values) > 1:
                    error_axis.hlines(
                        np.median(values),
                        position + shift - 0.055,
                        position + shift + 0.055,
                        color=METHOD_COLORS[method],
                        linewidth=2.0,
                    )
        error_axis.set_yscale("log")
        error_axis.set_xticks([0, 1], list(metric_positions))
        error_axis.set_ylabel("Rollout NRMSE")
        error_axis.set_title("b  Predictive accuracy", loc="left", fontweight="bold")
        error_axis.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.6)

        fig.suptitle(
            f"Brusselator feature recovery  |  {_campaign_label(records, expected_seeds)}",
            x=0.5,
            y=1.02,
            fontsize=10,
        )
        fig.tight_layout(w_pad=2.2)
        _save_figure(fig, output_stem)


def plot_robustness(
    output_stem: Path,
    records: list[FeatureRecoveryRecord],
    config: dict[str, Any],
) -> None:
    """Plot optimizer failure fraction and runtime by seed."""
    methods = [
        method
        for method in config["experiment"]["methods"]
        if _method_records(records, method)
    ]
    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
        for method in methods:
            method_records = _method_records(records, method)
            seeds = [record.seed for record in method_records]
            axes[0].plot(
                seeds,
                [record.failed_candidate_fraction for record in method_records],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method],
            )
            axes[1].plot(
                seeds,
                [record.wall_time_seconds / 60.0 for record in method_records],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method],
            )
        axes[0].set_title(
            "a  Invalid candidate fraction", loc="left", fontweight="bold"
        )
        axes[0].set_ylabel("Failed evaluations / total")
        axes[0].set_ylim(bottom=0)
        axes[1].set_title("b  Runtime", loc="left", fontweight="bold")
        axes[1].set_ylabel("Wall time (min)")
        axes[1].set_ylim(bottom=0)
        for axis in axes:
            axis.set_xlabel("Seed")
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        axes[0].legend(loc="best")
        fig.tight_layout(w_pad=2.2)
        _save_figure(fig, output_stem)


def plot_pareto(
    output_stem: Path,
    payloads: list[dict[str, Any]],
) -> None:
    """Plot every available two-objective Pareto front and selected point."""
    with publication_style():
        fig, axis = plt.subplots(figsize=(3.55, 2.8))
        for payload in payloads:
            method = payload["job"]["method"]
            seed = int(payload["job"]["data_seed"])
            points = []
            selected_point = None
            for entry in payload["pareto"]:
                objectives = entry["objectives"]
                values = list(objectives.values())
                if len(values) != 2 or not all(
                    _finite_number(value) for value in values
                ):
                    continue
                point = (float(values[0]), float(values[1]))
                points.append(point)
                if entry["index"] == payload["selected_index"]:
                    selected_point = point
            if not points:
                continue
            points_array = np.asarray(sorted(points))
            axis.plot(
                points_array[:, 0],
                points_array[:, 1],
                color=METHOD_COLORS[method],
                alpha=0.35,
                linewidth=0.8,
            )
            axis.scatter(
                points_array[:, 0],
                points_array[:, 1],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                s=16,
                alpha=0.55,
                label=f"{METHOD_LABELS[method]}, seed {seed}",
            )
            if selected_point is not None:
                axis.scatter(
                    [selected_point[0]],
                    [selected_point[1]],
                    facecolors="none",
                    edgecolors="#111111",
                    marker="o",
                    s=58,
                    linewidths=1.2,
                    zorder=4,
                )
        axis.set_xscale("log")
        axis.set_xlabel("Training trajectory MSE")
        axis.set_ylabel("Active coefficients")
        axis.set_title("Pareto fronts; selected models circled", loc="left")
        axis.grid(color="#D9D9D9", linewidth=0.6)
        axis.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, output_stem)


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def write_quality_report(
    path: Path,
    records: list[FeatureRecoveryRecord],
    config: dict[str, Any],
    raw_directory: Path,
) -> dict[str, Any]:
    expected_methods = list(config["experiment"]["methods"])
    expected_seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    observed = {
        method: sorted(record.seed for record in records if record.method == method)
        for method in expected_methods
    }
    missing = {
        method: sorted(set(expected_seeds) - set(observed[method]))
        for method in expected_methods
    }
    dirty_jobs = [record.job_id for record in records if record.git_dirty]
    all_json = list(raw_directory.glob("*.json"))
    report = {
        "publication_ready": not any(missing.values()) and not dirty_jobs,
        "valid_production_results": len(records),
        "json_files_scanned": len(all_json),
        "observed_seeds": observed,
        "missing_seeds": missing,
        "dirty_provenance_jobs": dirty_jobs,
        "warnings": [],
    }
    if any(missing.values()):
        report["warnings"].append("The production campaign is incomplete.")
    if dirty_jobs:
        report["warnings"].append(
            "Some production results were generated from a dirty worktree."
        )
    if len(all_json) != len(records):
        report["warnings"].append(
            "Smoke, failed, or malformed JSON files were excluded."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=repo_root / "benchmarks/results/raw/feature_recovery",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "benchmarks/configs/feature_recovery.toml",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repo_root / "benchmarks/results",
    )
    args = parser.parse_args()

    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    records, payloads = collect_records(args.raw_directory, config)
    if not records:
        raise SystemExit("No valid production feature-recovery results found.")

    aggregate_directory = args.output_directory / "aggregated"
    figure_directory = args.output_directory / "figures"
    write_summary_csv(aggregate_directory / "feature_recovery_runs.csv", records)
    report = write_quality_report(
        aggregate_directory / "feature_recovery_quality.json",
        records,
        config,
        args.raw_directory,
    )
    plot_summary(figure_directory / "feature_recovery_summary", records, config)
    plot_robustness(figure_directory / "feature_recovery_robustness", records, config)
    plot_pareto(figure_directory / "feature_recovery_pareto", payloads)

    print(f"processed {len(records)} valid production results")
    print(f"publication ready: {report['publication_ready']}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    print(f"wrote {aggregate_directory}")
    print(f"wrote {figure_directory}")


if __name__ == "__main__":
    main()
