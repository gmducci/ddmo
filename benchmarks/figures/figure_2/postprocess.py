"""Aggregate Figure 2 runs and create noise-robustness previews."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import tomllib

from benchmarks.common.goodwin import true_coefficient_matrix
from benchmarks.common.style import publication_style

plt.switch_backend("Agg")

METHODS = ["strong", "weak_fixed", "weak_optimized", "weak_optimized_sg"]
METHOD_LABELS = {
    "strong": "Strong",
    "weak_fixed": "Weak, fixed",
    "weak_optimized": "Weak, optimized",
    "weak_optimized_sg": "Weak, optimized + SG",
}
METHOD_COLORS = {
    "strong": "#3B4CC0",
    "weak_fixed": "#00876C",
    "weak_optimized": "#B40426",
    "weak_optimized_sg": "#E07A1F",
}
METHOD_MARKERS = {
    "strong": "s",
    "weak_fixed": "^",
    "weak_optimized": "o",
    "weak_optimized_sg": "D",
}
STATE_NAMES = ["x", "y", "z"]
STATE_COLORS = ["#3B4CC0", "#00876C", "#B40426"]
COEFFICIENT_SPECS = [
    (r"$\dot{x}$: Hill$(z)$", "hill_z", 0, "k1", 1.0),
    (r"$\dot{y}$: $x$", "x", 1, "k3", 1.0),
    (r"$\dot{z}$: $y$", "y", 2, "k5", 1.0),
    (r"$\dot{x}$: $x$", "x", 0, "k2", -1.0),
    (r"$\dot{y}$: $y$", "y", 1, "k4", -1.0),
    (r"$\dot{z}$: $z$", "z", 2, "k6", -1.0),
]


@dataclass(frozen=True)
class NoiseRecord:
    source_file: str
    job_id: str
    method: str
    noise_level: float
    seed: int
    validation_nrmse: float
    test_nrmse: float
    test_rollout_failed: bool
    coefficient_relative_l2_error: float
    support_accuracy: float
    support_precision: float
    support_recall: float
    support_f1: float
    false_positive_coefficients: int
    false_negative_coefficients: int
    failed_candidate_fraction: float
    wall_time_seconds: float
    degree: float | None
    window_length: float | None
    stride: float | None
    git_commit: str
    git_dirty: bool


def _number(value: Any) -> float:
    return float(value) if value is not None else math.nan


def collect_records(
    raw_directory: Path, expected_experiment: str = "weak_noise_poly2"
) -> list[NoiseRecord]:
    records = []
    for path in sorted(raw_directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete" or payload.get("mode") != "production":
            continue
        job = payload.get("job", {})
        if job.get("experiment") != expected_experiment:
            continue
        selected = payload["selected"]
        variables = selected["variables"]
        coefficient_metrics = selected["coefficient_metrics"]
        evaluations = int(payload["actual_candidate_evaluations"])
        failed = int(payload["failed_candidate_count"])
        test_nrmse = _number(payload["test"].get("rollout_nrmse"))
        records.append(
            NoiseRecord(
                source_file=str(path),
                job_id=str(job["job_id"]),
                method=str(job["method"]),
                noise_level=float(job["noise_level"]),
                seed=int(job["data_seed"]),
                validation_nrmse=_number(selected["validation"].get("rollout_nrmse")),
                test_nrmse=test_nrmse,
                test_rollout_failed=not math.isfinite(test_nrmse),
                coefficient_relative_l2_error=_number(
                    coefficient_metrics.get("relative_l2_error")
                ),
                support_accuracy=_number(coefficient_metrics.get("support_accuracy")),
                support_precision=_number(coefficient_metrics.get("support_precision")),
                support_recall=_number(coefficient_metrics.get("support_recall")),
                support_f1=_number(coefficient_metrics.get("support_f1")),
                false_positive_coefficients=int(
                    coefficient_metrics.get("false_positive_coefficients", 0)
                ),
                false_negative_coefficients=int(
                    coefficient_metrics.get("false_negative_coefficients", 0)
                ),
                failed_candidate_fraction=failed / evaluations
                if evaluations
                else math.nan,
                wall_time_seconds=float(payload["wall_time_seconds"]),
                degree=variables.get("degree"),
                window_length=variables.get("window_length"),
                stride=variables.get("stride"),
                git_commit=str(payload["provenance"]["git_commit"]),
                git_dirty=bool(payload["provenance"]["git_dirty"]),
            )
        )
    return records


def write_csv(path: Path, records: list[NoiseRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _values(
    records: list[NoiseRecord], method: str, noise: float, attribute: str
) -> np.ndarray:
    return np.asarray(
        [
            getattr(record, attribute)
            for record in records
            if record.method == method and np.isclose(record.noise_level, noise)
        ],
        dtype=float,
    )


def _plot_metric(
    axis: plt.Axes,
    records: list[NoiseRecord],
    noise_levels: list[float],
    attribute: str,
    ylabel: str,
    *,
    logarithmic: bool,
) -> None:
    for method in METHODS:
        medians = []
        plotted_noise = []
        for noise in noise_levels:
            values = _values(records, method, noise, attribute)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            offsets = np.linspace(-0.0025, 0.0025, len(finite))
            axis.scatter(
                noise + offsets,
                finite,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                s=25,
                alpha=0.55,
                zorder=3,
            )
            plotted_noise.append(noise)
            medians.append(float(np.median(finite)))
        if plotted_noise:
            axis.plot(
                plotted_noise,
                medians,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method],
                zorder=4,
            )
    if logarithmic:
        axis.set_yscale("log")
    axis.set_xlabel("Noise level")
    axis.set_ylabel(ylabel)
    axis.set_xticks(noise_levels, [f"{100 * value:.0f}%" for value in noise_levels])
    axis.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.6)


def plot_summary(
    output_stem: Path,
    records: list[NoiseRecord],
    config: dict[str, Any],
) -> None:
    noise_levels = [float(value) for value in config["experiment"]["noise_levels"]]
    expected = (
        len(config["experiment"]["methods"])
        * len(noise_levels)
        * len(config["experiment"]["seeds"])
    )
    clean = sum(not record.git_dirty for record in records)
    state = "FINAL" if len(records) == expected and clean == expected else "PREVIEW"
    with publication_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
        _plot_metric(
            axes[0],
            records,
            noise_levels,
            "test_nrmse",
            "Test rollout NRMSE",
            logarithmic=True,
        )
        _plot_metric(
            axes[1],
            records,
            noise_levels,
            "coefficient_relative_l2_error",
            "Coefficient relative error",
            logarithmic=True,
        )
        _plot_metric(
            axes[2],
            records,
            noise_levels,
            "false_positive_coefficients",
            "False-positive coefficients",
            logarithmic=False,
        )
        for label, axis in zip("abc", axes):
            axis.set_title(label, loc="left", fontweight="bold")
        axes[0].legend(loc="best")
        fig.suptitle(
            f"Goodwin noise robustness  |  {state} — {len(records)}/{expected} jobs",
            fontsize=10,
            y=1.03,
        )
        fig.tight_layout(w_pad=1.5)
        _save(fig, output_stem)


def plot_reconstructed_dynamics(
    output_stem: Path,
    records: list[NoiseRecord],
    config: dict[str, Any],
) -> None:
    """Plot a median-representative SG+weak rollout at each noise level."""
    method = "weak_optimized_sg"
    representatives = []
    for noise in config["experiment"]["noise_levels"]:
        candidates = [
            record
            for record in records
            if record.method == method
            and np.isclose(record.noise_level, float(noise))
            and math.isfinite(record.test_nrmse)
        ]
        if not candidates:
            continue
        median = float(np.median([record.test_nrmse for record in candidates]))
        representatives.append(
            min(candidates, key=lambda record: abs(record.test_nrmse - median))
        )
    if not representatives:
        return

    with publication_style():
        fig, axes = plt.subplots(
            len(representatives),
            len(STATE_NAMES),
            figsize=(7.2, 7.6),
            sharex=True,
            squeeze=False,
        )
        for row, record in enumerate(representatives):
            artifact = Path(record.source_file).with_suffix(".npz")
            arrays = np.load(artifact)
            time = arrays["time"]
            clean = arrays["test_clean"][0]
            noisy = arrays["test_noisy"][0]
            prediction = arrays["test_prediction"][0]
            for column, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
                axis = axes[row, column]
                axis.plot(
                    time,
                    clean[:, column],
                    color="#222222",
                    linewidth=1.25,
                    label="Clean ground truth",
                )
                axis.scatter(
                    time[::10],
                    noisy[::10, column],
                    color=color,
                    s=5,
                    alpha=0.25,
                    edgecolors="none",
                    label="Noisy observations",
                )
                axis.plot(
                    time,
                    prediction[:, column],
                    color=color,
                    linewidth=1.3,
                    linestyle="--",
                    label="Reconstructed rollout",
                )
                axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
                if row == 0:
                    axis.set_title(name, fontweight="bold")
                if column == 0:
                    axis.set_ylabel(
                        f"{100 * record.noise_level:.0f}%\nseed {record.seed}",
                        rotation=0,
                        ha="right",
                        va="center",
                    )
                if column == 2:
                    axis.text(
                        0.98,
                        0.92,
                        f"NRMSE {record.test_nrmse:.3g}",
                        ha="right",
                        va="top",
                        transform=axis.transAxes,
                        fontsize=7,
                    )
                if row == len(representatives) - 1:
                    axis.set_xlabel("Time")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.985),
        )
        fig.suptitle(
            "Representative held-out Goodwin reconstructions — optimized weak + SG",
            fontsize=10,
            y=1.015,
        )
        fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.965), h_pad=0.6, w_pad=0.7)
        _save(fig, output_stem)


def _selected_coefficient(
    record: NoiseRecord, feature_key: str, state_index: int
) -> float:
    payload = json.loads(Path(record.source_file).read_text(encoding="utf-8"))
    selected = payload["selected"]
    names = list(selected["feature_names"])
    if feature_key == "hill_z":
        feature_index = next(
            index for index, name in enumerate(names) if name.startswith("1/(1+z^")
        )
    else:
        feature_index = names.index(feature_key)
    coefficients = np.asarray(selected["coefficients"], dtype=float)
    return float(coefficients[feature_index, state_index])


def plot_coefficient_recovery(
    output_stem: Path,
    records: list[NoiseRecord],
    config: dict[str, Any],
) -> None:
    """Compare recovered active coefficients with the Goodwin ground truth."""
    method_records = [
        record for record in records if record.method == "weak_optimized_sg"
    ]
    if not method_records:
        return
    noise_levels = [float(value) for value in config["experiment"]["noise_levels"]]
    truth = config["truth"]
    with publication_style():
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=True)
        for axis, (title, feature, state_index, truth_key, sign) in zip(
            axes.flat, COEFFICIENT_SPECS
        ):
            ground_truth = sign * float(truth[truth_key])
            medians = []
            for noise_index, noise in enumerate(noise_levels):
                group = [
                    record
                    for record in method_records
                    if np.isclose(record.noise_level, noise)
                ]
                values = np.asarray(
                    [
                        _selected_coefficient(record, feature, state_index)
                        for record in group
                    ],
                    dtype=float,
                )
                offsets = np.linspace(-0.0025, 0.0025, len(values))
                axis.scatter(
                    noise + offsets,
                    values,
                    color="#E07A1F",
                    s=22,
                    alpha=0.45,
                    zorder=3,
                    label="Selected runs" if noise_index == 0 else "_nolegend_",
                )
                medians.append(float(np.median(values)))
            axis.plot(
                noise_levels,
                medians,
                color="#E07A1F",
                marker="D",
                markersize=4,
                linewidth=1.2,
                zorder=4,
                label="Median estimate",
            )
            axis.axhline(
                ground_truth,
                color="#222222",
                linestyle="--",
                linewidth=1.1,
                label=f"Ground truth ({ground_truth:g})",
            )
            axis.set_title(title, loc="left", fontweight="bold")
            axis.set_xticks(
                noise_levels, [f"{100 * noise:.0f}%" for noise in noise_levels]
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        for axis in axes[1]:
            axis.set_xlabel("Noise level")
        for axis in axes[:, 0]:
            axis.set_ylabel("Coefficient")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.985),
        )
        fig.suptitle(
            "Goodwin coefficient recovery — optimized weak + SG",
            fontsize=10,
            y=1.02,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), h_pad=1.0, w_pad=1.1)
        _save(fig, output_stem)


def _coefficient_payload(record: NoiseRecord) -> tuple[list[str], np.ndarray]:
    payload = json.loads(Path(record.source_file).read_text(encoding="utf-8"))
    selected = payload["selected"]
    return (
        list(selected["feature_names"]),
        np.asarray(selected["coefficients"], dtype=float),
    )


def write_full_library_csv(
    path: Path, records: list[NoiseRecord], config: dict[str, Any]
) -> None:
    """Write all 36 coefficient comparisons, including zero distractors."""
    method_records = [
        record for record in records if record.method == "weak_optimized_sg"
    ]
    feature_names, _ = _coefficient_payload(method_records[0])
    truth = true_coefficient_matrix(feature_names, config)
    rows = []
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    for noise in config["experiment"]["noise_levels"]:
        matrices = []
        for record in method_records:
            if not np.isclose(record.noise_level, float(noise)):
                continue
            names, matrix = _coefficient_payload(record)
            if names != feature_names:
                raise ValueError("Inconsistent feature order in Figure 2 results")
            matrices.append(matrix)
        stacked = np.stack(matrices)
        for feature_index, feature in enumerate(feature_names):
            for state_index, state in enumerate(STATE_NAMES):
                values = stacked[:, feature_index, state_index]
                rows.append(
                    {
                        "noise_level": float(noise),
                        "feature": feature,
                        "equation": f"d{state}/dt",
                        "ground_truth": float(truth[feature_index, state_index]),
                        "median_estimate": float(np.median(values)),
                        "q25_estimate": float(np.quantile(values, 0.25)),
                        "q75_estimate": float(np.quantile(values, 0.75)),
                        "selection_frequency": float(
                            np.mean(np.abs(values) > tolerance)
                        ),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_full_library_coefficients(
    output_stem: Path, records: list[NoiseRecord], config: dict[str, Any]
) -> None:
    """Show the complete candidate coefficient matrix versus ground truth."""
    method_records = [
        record for record in records if record.method == "weak_optimized_sg"
    ]
    if not method_records:
        return
    feature_names, _ = _coefficient_payload(method_records[0])
    truth = true_coefficient_matrix(feature_names, config)
    matrices = [("Ground truth", truth)]
    for noise in config["experiment"]["noise_levels"]:
        estimates = []
        for record in method_records:
            if np.isclose(record.noise_level, float(noise)):
                names, matrix = _coefficient_payload(record)
                if names != feature_names:
                    raise ValueError("Inconsistent feature order in Figure 2 results")
                estimates.append(matrix)
        matrices.append((f"{100 * float(noise):.0f}% noise", np.median(estimates, axis=0)))

    max_abs = max(float(np.max(np.abs(matrix))) for _, matrix in matrices)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    with publication_style():
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 7.0), squeeze=False)
        image = None
        for index, (axis, (title, matrix)) in enumerate(zip(axes.flat, matrices)):
            image = axis.imshow(
                matrix,
                cmap="RdBu_r",
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            axis.set_title(title, fontweight="bold")
            axis.set_xticks(range(3), [r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"])
            axis.set_yticks(range(len(feature_names)))
            axis.set_yticklabels(feature_names if index % 3 == 0 else [])
            axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.4)
            axis.tick_params(which="minor", bottom=False, left=False)
        colorbar_axis = fig.add_axes((0.20, 0.045, 0.62, 0.025))
        colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("Coefficient (median across five seeds)")
        fig.suptitle(
            "Complete degree-2 + Hill library — optimized weak + SG",
            fontsize=10,
            y=0.995,
        )
        fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.15, hspace=0.28, wspace=0.14)
        _save(fig, output_stem)


def write_quality(
    path: Path, records: list[NoiseRecord], config: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        (method, float(noise), int(seed))
        for method in config["experiment"]["methods"]
        for noise in config["experiment"]["noise_levels"]
        for seed in config["experiment"]["seeds"]
    }
    observed = {(r.method, r.noise_level, r.seed) for r in records}
    report = {
        "publication_ready": observed == expected
        and not any(record.git_dirty for record in records),
        "valid_production_results": len(records),
        "expected_results": len(expected),
        "missing_jobs": [list(item) for item in sorted(expected - observed)],
        "dirty_provenance_jobs": [r.job_id for r in records if r.git_dirty],
        "failed_test_rollout_jobs": [
            r.job_id for r in records if r.test_rollout_failed
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    with (Path(__file__).with_name("config.toml")).open("rb") as stream:
        config = tomllib.load(stream)
    experiment_name = str(config["experiment"]["name"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=repo_root / "benchmarks/results/raw" / experiment_name,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repo_root / "benchmarks/results",
    )
    args = parser.parse_args()
    records = collect_records(args.raw_directory, experiment_name)
    if not records:
        raise SystemExit("No Figure 2 production results found.")
    aggregate = args.output_directory / "aggregated"
    figures = args.output_directory / "figures"
    write_csv(aggregate / "figure_2_runs.csv", records)
    write_full_library_csv(
        aggregate / "figure_2_full_library_coefficients.csv", records, config
    )
    report = write_quality(aggregate / "figure_2_quality.json", records, config)
    plot_summary(figures / "figure_2_noise_robustness", records, config)
    plot_reconstructed_dynamics(
        figures / "figure_2_reconstructed_dynamics", records, config
    )
    plot_coefficient_recovery(
        figures / "figure_2_coefficient_recovery", records, config
    )
    plot_full_library_coefficients(
        figures / "figure_2_full_library_coefficients", records, config
    )
    print(f"processed {len(records)} production results")
    print(f"publication ready: {report['publication_ready']}")
    print(f"wrote {figures / 'figure_2_noise_robustness.png'}")


if __name__ == "__main__":
    main()
