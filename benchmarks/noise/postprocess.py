"""Figure 1 and Figure S1: noise robustness and all-library coefficient recovery."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from benchmarks.common.data import result_data
from benchmarks.common.goodwin import true_coefficient_matrix
from benchmarks.common.style import publication_style

plt.switch_backend("Agg")

METHODS = ["strong", "weak_optimized", "weak_optimized_sg"]
METHOD_LABELS = {
    "strong": "Strong + SG",
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
            axis.scatter(
                np.full(finite.shape, noise),
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
                zorder=5 if method == "weak_optimized" else 4,
                linestyle=(0, (4, 3)) if method == "weak_optimized" else "-",
                markerfacecolor="none" if method == "weak_optimized" else METHOD_COLORS[method],
                markeredgewidth=1.2 if method == "weak_optimized" else 1.0,
                markersize=6.0 if method == "weak_optimized" else 5.5,
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
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(METHODS),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82), w_pad=1.5)
        for label, axis in zip("abc", axes):
            bounds = axis.get_position()
            fig.text(
                bounds.x0,
                bounds.y1 + 0.025,
                label,
                ha="left",
                va="bottom",
                fontweight="bold",
            )
        _save(fig, output_stem)


def _coefficient_payload(record: NoiseRecord) -> tuple[list[str], np.ndarray]:
    payload = json.loads(Path(record.source_file).read_text(encoding="utf-8"))
    selected = payload["selected"]
    return (
        list(selected["feature_names"]),
        np.asarray(selected["coefficients"], dtype=float),
    )


def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def feature_label(name):
    if name.startswith("1/(1+"):
        return r"$h(" + name.removeprefix("1/(1+").split("^")[0] + r")$"
    return "$" + name.replace("*", r"\,").replace("^2", "^{2}") + "$"


def plot_coefficients(stem, records, config):
    feature_names, _ = _coefficient_payload(records[0])
    truth = true_coefficient_matrix(feature_names, config)
    noises = np.asarray(config["experiment"]["noise_levels"], dtype=float)
    groups = {}
    for method in METHODS:
        for noise in noises:
            group = sorted([r for r in records if r.method == method and np.isclose(r.noise_level, noise)], key=lambda r: r.seed)
            matrices = []
            for record in group:
                names, matrix = _coefficient_payload(record)
                if names != feature_names or matrix.shape != (12, 3):
                    raise ValueError("Inconsistent noise coefficient library")
                matrices.append(matrix)
            groups[method, float(noise)] = np.stack(matrices)
    with publication_style():
        fig, axes = plt.subplots(12, 3, figsize=(10.4, 17.6), sharex=True)
        fig.subplots_adjust(left=.105, right=.985, bottom=.04, top=.953,
                            wspace=.36, hspace=.30)
        for i, name in enumerate(feature_names):
            for j in range(3):
                axis = axes[i, j]
                axis.axhline(truth[i, j], color="#292929", linestyle=(0, (2, 2)),
                             linewidth=.9, zorder=2)
                all_values = [truth[i, j]]
                for method in METHODS:
                    color = METHOD_COLORS[method]
                    marker = METHOD_MARKERS[method]
                    medians = []
                    for noise in noises:
                        values = groups[method, float(noise)][:, i, j]
                        all_values.extend(values)
                        axis.scatter(noise + np.linspace(-.0025, .0025, len(values)), values,
                                     color=color, marker=marker, s=9, alpha=.28,
                                     linewidths=.4, zorder=3)
                        medians.append(float(np.median(values)))
                    red = method == "weak_optimized"
                    axis.plot(noises, medians, color=color, marker=marker,
                              linestyle=(0, (4, 3)) if red else "-",
                              markerfacecolor="none" if red else color,
                              markersize=4.3 if red else 3.3,
                              markeredgewidth=.9, linewidth=1.1,
                              zorder=5 if red else 4)
                lo, hi = min(all_values), max(all_values)
                span = hi - lo
                pad = .12 * span if span else max(.02, abs(lo) * .1)
                axis.set_ylim(lo - pad, hi + pad)
                axis.set_xlim(-.009, .209)
                axis.set_xticks(noises, [f"{100*n:.0f}%" for n in noises])
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useOffset=False)
                axis.yaxis.get_offset_text().set_fontsize(6.5)
                axis.tick_params(labelsize=7, length=2.5, pad=2)
                axis.grid(axis="y", color="#E4E4E4", linewidth=.45)
                if j == 0:
                    axis.set_ylabel(feature_label(name), fontsize=10, rotation=0,
                                    ha="right", va="center", labelpad=22)
        # Equation and feature labels identify axes; there is no figure title
        # and no per-panel title.
        for j, state in enumerate(STATE_NAMES):
            bounds = axes[0, j].get_position()
            fig.text((bounds.x0 + bounds.x1)/2, .958, rf"$\dot{{{state}}}$",
                     ha="center", va="bottom", fontsize=12)
        handles = []
        for method in METHODS:
            red = method == "weak_optimized"
            handles.append(Line2D([], [], color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], linestyle=(0, (4, 3)) if red else "-",
                markerfacecolor="none" if red else METHOD_COLORS[method],
                markersize=5, label=METHOD_LABELS[method]))
        handles.append(Line2D([], [], color="#292929", linestyle=(0, (2, 2)), label="Ground truth"))
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(.53, .996), fontsize=9, handlelength=3)
        fig.text(.022, .53, "Library term / coefficient", rotation=90,
                 ha="center", va="center", fontsize=10)
        fig.text(.545, .012, "Noise level", ha="center", va="bottom", fontsize=10)
        assert fig._suptitle is None
        assert all(not axis.get_title() for axis in axes.flat)
        _save(fig, stem)


def render(raw, config, output):
    records = collect_records(raw, config["experiment"]["name"])
    output.mkdir(parents=True, exist_ok=True)
    plot_summary(output / "figure_1_noise_robustness", records, config)
    plot_coefficients(output / "figure_s1_full_coefficient_recovery", records, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path(".local/benchmark_plots/noise"))
    args = parser.parse_args()
    with result_data("noise", args.raw_directory) as (config, raw):
        render(raw, config, args.output_directory)
    print(f"Wrote Figure 1 and S1 to {args.output_directory}")


if __name__ == "__main__":
    main()
