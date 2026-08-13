"""Aggregate Figure 3 runs and plot the joint-optimization demonstration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import tomllib

from benchmarks.common.goodwin import true_coefficient_matrix
from benchmarks.common.style import publication_style

plt.switch_backend("Agg")
STATE_NAMES = ["x", "y", "z"]
STATE_COLORS = ["#3B4CC0", "#00876C", "#B40426"]


def _canonical_feature(name: str) -> str:
    if name.startswith("1/(1+"):
        variable = name.removeprefix("1/(1+").split("^")[0]
        return f"Hill({variable})"
    if name.startswith("exp("):
        variable = name.rsplit("*", 1)[-1].removesuffix(")")
        return f"exp({variable})"
    return name


def _family_selected(payload: dict[str, Any], family: str) -> bool:
    selected = payload["selected"]
    tolerance = 1e-6
    for name, row in zip(selected["feature_names"], selected["coefficients"]):
        if _canonical_feature(name).startswith(family) and np.any(
            np.abs(np.asarray(row, dtype=float)) > tolerance
        ):
            return True
    return False


def collect_payloads(
    raw_directory: Path, expected_experiment: str
) -> list[tuple[dict[str, Any], Path]]:
    payloads = []
    for path in sorted(raw_directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete" or payload.get("mode") != "production":
            continue
        if payload.get("job", {}).get("experiment") != expected_experiment:
            continue
        artifact = path.with_name(payload["trajectory_artifact"])
        if artifact.exists():
            payloads.append((payload, artifact))
    return payloads


def _summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    variables = payload["selected"]["variables"]
    recovery = payload["exponent_recovery"]
    test_nrmse = payload["test"].get("rollout_nrmse")
    evaluations = int(payload["actual_candidate_evaluations"])
    hill_selected = _family_selected(payload, "Hill(")
    exponential_selected = _family_selected(payload, "exp(")
    return {
        "job_id": payload["job"]["job_id"],
        "seed": payload["job"]["data_seed"],
        "gamma_1": recovery["recovered"],
        "gamma_1_relative_error": recovery["relative_error"],
        "gamma_1_identifiable": hill_selected,
        "gamma_2": variables["gamma_2"],
        "exponential_family_selected": exponential_selected,
        "degree": variables["degree"],
        "window_length": variables["window_length"],
        "stride": variables["stride"],
        "threshold": variables["threshold"],
        "alpha": variables["alpha"],
        "validation_nrmse": payload["selected"]["validation"]["rollout_nrmse"],
        "test_nrmse": test_nrmse,
        "coefficient_relative_l2_error": payload["selected"]["coefficient_metrics"][
            "relative_l2_error"
        ],
        "support_accuracy": payload["selected"]["coefficient_metrics"][
            "support_accuracy"
        ],
        "support_f1": payload["selected"]["coefficient_metrics"]["support_f1"],
        "false_positive_coefficients": payload["selected"]["coefficient_metrics"][
            "false_positive_coefficients"
        ],
        "failed_candidate_fraction": payload["failed_candidate_count"] / evaluations,
        "wall_time_seconds": payload["wall_time_seconds"],
        "git_commit": payload["provenance"]["git_commit"],
        "git_dirty": payload["provenance"]["git_dirty"],
    }


def write_csv(path: Path, payloads: list[tuple[dict[str, Any], Path]]) -> None:
    rows = [_summary_row(payload) for payload, _ in payloads]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_joint_result(
    output_stem: Path,
    payload: dict[str, Any],
    artifact_path: Path,
    payloads: list[tuple[dict[str, Any], Path]],
    expected_jobs: int,
) -> None:
    arrays = np.load(artifact_path)
    time = arrays["time"]
    clean = arrays["test_clean"][0]
    noisy = arrays["test_noisy"][0]
    prediction = arrays["test_prediction"][0]
    recovery = payload["exponent_recovery"]
    gamma_truth = float(recovery["truth"])
    hill_payloads = [item for item in payloads if _family_selected(item[0], "Hill(")]
    exponential_payloads = [
        item for item in payloads if _family_selected(item[0], "exp(")
    ]
    recovered_gamma_1 = np.asarray(
        [item[0]["selected"]["variables"]["gamma_1"] for item in hill_payloads],
        dtype=float,
    )
    recovered_gamma_2 = np.asarray(
        [item[0]["selected"]["variables"]["gamma_2"] for item in exponential_payloads],
        dtype=float,
    )
    median_gamma_1 = float(np.median(recovered_gamma_1))
    median_gamma_2 = float(np.median(recovered_gamma_2))
    state = (
        "PREVIEW"
        if payload["provenance"]["git_dirty"] or expected_jobs > 1
        else "FINAL"
    )
    with publication_style():
        fig = plt.figure(figsize=(7.2, 5.2))
        grid = fig.add_gridspec(
            3, 2, width_ratios=[2.15, 1.55], hspace=0.28, wspace=0.34
        )
        for index, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
            axis = fig.add_subplot(grid[index, 0])
            axis.plot(time, clean[:, index], color="#222222", label="Clean reference")
            axis.scatter(
                time[::10],
                noisy[::10, index],
                color=color,
                s=8,
                alpha=0.35,
                label="Noisy observations",
            )
            if np.any(np.isfinite(prediction[:, index])):
                axis.plot(
                    time,
                    prediction[:, index],
                    color=color,
                    linestyle="--",
                    label="Identified rollout",
                )
            axis.set_ylabel(name)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            if index < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Time")
            if index == 0:
                handles, labels = axis.get_legend_handles_labels()

        library_axis = fig.add_subplot(grid[0:2, 1])
        library_input = np.linspace(0.0, 2.0, 500)
        truth_library = 1.0 / (1.0 + library_input**gamma_truth)
        selected_library = 1.0 / (1.0 + library_input**median_gamma_1)
        library_axis.plot(
            library_input,
            truth_library,
            color="#222222",
            linewidth=1.6,
            label=rf"Ground truth ($\gamma={gamma_truth:.3g}$)",
        )
        for gamma in recovered_gamma_1:
            library_axis.plot(
                library_input,
                1.0 / (1.0 + library_input**gamma),
                color=STATE_COLORS[2],
                linewidth=0.7,
                alpha=0.22,
            )
        library_axis.plot(
            library_input,
            selected_library,
            color=STATE_COLORS[2],
            linewidth=1.6,
            linestyle="--",
            label=rf"Median optimized ($\gamma_1={median_gamma_1:.3g}$)",
        )
        library_axis.axvline(1.0, color="#BDBDBD", linewidth=0.7, zorder=0)
        library_axis.set(
            xlabel="Library input",
            ylabel=r"$1/(1+s^\gamma)$",
            xlim=(0.0, 2.0),
            ylim=(-0.03, 1.03),
        )
        for gamma in recovered_gamma_2:
            library_axis.plot(
                library_input,
                np.exp(-gamma * library_input),
                color=STATE_COLORS[0],
                linewidth=0.7,
                alpha=0.22,
            )
        library_axis.plot(
            library_input,
            np.exp(-median_gamma_2 * library_input),
            color=STATE_COLORS[0],
            linewidth=1.35,
            linestyle=":",
            label=rf"Selected nuisance exp. (median $\gamma_2={median_gamma_2:.3g}$)",
        )
        library_axis.set_title("Optimized nonlinear candidates", loc="left")
        library_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        library_axis.legend(loc="upper right", frameon=False, fontsize=7)
        parameter_axis = fig.add_subplot(grid[2, 1])
        parameter_axis.axis("off")
        metrics = payload["selected"]["coefficient_metrics"]
        parameter_axis.text(
            0.0,
            0.78,
            "Conditional library-parameter recovery\n"
            + rf"Hill selected {len(hill_payloads)}/5; $\gamma_1$ median "
            + rf"{median_gamma_1:.4g}, range "
            + rf"[{np.min(recovered_gamma_1):.4g}, {np.max(recovered_gamma_1):.4g}]"
            + "\n"
            + rf"Exponential selected {len(exponential_payloads)}/5; $\gamma_2$ range "
            + rf"[{np.min(recovered_gamma_2):.4g}, {np.max(recovered_gamma_2):.4g}]"
            + "\n"
            + rf"Representative seed: {payload['job']['data_seed']}, "
            + rf"support F1={metrics['support_f1']:.3g}, "
            + rf"false positives={metrics['false_positive_coefficients']}",
            ha="left",
            va="top",
            fontsize=7.2,
            linespacing=1.45,
        )
        test_value = payload["test"].get("rollout_nrmse")
        test_text = "failed" if test_value is None else f"{float(test_value):.3g}"
        fig.suptitle(
            f"Joint Goodwin optimization  |  {state}  |  test NRMSE {test_text}",
            fontsize=10,
            y=0.995,
        )
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.06, 0.955),
            ncol=3,
            frameon=False,
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"))
        fig.savefig(output_stem.with_suffix(".pdf"))
        plt.close(fig)


def write_coefficient_csv(
    path: Path, payloads: list[tuple[dict[str, Any], Path]], config: dict[str, Any]
) -> None:
    rows = []
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    for payload, _ in payloads:
        selected = payload["selected"]
        names = list(selected["feature_names"])
        coefficients = np.asarray(selected["coefficients"], dtype=float)
        truth = true_coefficient_matrix(names, config)
        for feature_index, feature in enumerate(names):
            for state_index, state in enumerate(STATE_NAMES):
                estimate = float(coefficients[feature_index, state_index])
                rows.append(
                    {
                        "seed": int(payload["job"]["data_seed"]),
                        "feature": _canonical_feature(feature),
                        "equation": f"d{state}/dt",
                        "ground_truth": float(truth[feature_index, state_index]),
                        "estimate": estimate,
                        "selected": abs(estimate) > tolerance,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_full_library(
    output_stem: Path,
    payloads: list[tuple[dict[str, Any], Path]],
    config: dict[str, Any],
) -> None:
    first = payloads[0][0]["selected"]
    reference_names = list(first["feature_names"])
    names = [_canonical_feature(name) for name in reference_names]
    estimates = []
    for payload, _ in payloads:
        selected = payload["selected"]
        current_names = [
            _canonical_feature(name) for name in selected["feature_names"]
        ]
        if set(current_names) != set(names):
            raise ValueError("Inconsistent Figure 3 feature order")
        matrix = np.asarray(selected["coefficients"], dtype=float)
        order = [current_names.index(name) for name in names]
        estimates.append(matrix[order])
    truth = true_coefficient_matrix(reference_names, config)
    median = np.median(estimates, axis=0)
    maximum = max(float(np.max(np.abs(truth))), float(np.max(np.abs(median))))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.9), sharey=True)
        image = None
        for axis, matrix, title in zip(
            axes, [truth, median], ["Ground truth", "Median selected model"]
        ):
            image = axis.imshow(
                matrix, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest"
            )
            axis.set_title(title, fontweight="bold")
            axis.set_xticks(range(3), [r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"])
            axis.set_yticks(range(len(names)), names)
            axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.4)
            axis.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.045, pad=0.10)
        colorbar.set_label("Coefficient")
        fig.suptitle("Figure 3 complete polynomial + Hill + exponential library", fontsize=10)
        fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.18, wspace=0.12)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"))
        fig.savefig(output_stem.with_suffix(".pdf"))
        plt.close(fig)


def plot_selection_frequency(
    output_stem: Path,
    payloads: list[tuple[dict[str, Any], Path]],
    config: dict[str, Any],
) -> None:
    reference_names = list(payloads[0][0]["selected"]["feature_names"])
    names = [_canonical_feature(name) for name in reference_names]
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    supports = []
    for payload, _ in payloads:
        selected = payload["selected"]
        current_names = [
            _canonical_feature(name) for name in selected["feature_names"]
        ]
        matrix = np.asarray(selected["coefficients"], dtype=float)
        order = [current_names.index(name) for name in names]
        supports.append(np.abs(matrix[order]) > tolerance)
    frequency = np.mean(supports, axis=0)
    with publication_style():
        fig, axis = plt.subplots(figsize=(4.4, 4.9))
        image = axis.imshow(
            frequency,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks(range(3), [r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"])
        axis.set_yticks(range(len(names)), names)
        for row, column in zip(*np.where(frequency > 0)):
            axis.text(
                column,
                row,
                f"{100 * frequency[row, column]:.0f}%",
                ha="center",
                va="center",
                color="white" if frequency[row, column] < 0.65 else "black",
                fontsize=7,
            )
        colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
        colorbar.set_label("Selection frequency across five seeds")
        axis.set_title("Figure 3 term-selection stability", fontweight="bold")
        fig.tight_layout()
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"))
        fig.savefig(output_stem.with_suffix(".pdf"))
        plt.close(fig)


def write_quality(
    path: Path,
    payloads: list[tuple[dict[str, Any], Path]],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_seeds = {int(seed) for seed in config["experiment"]["seeds"]}
    observed = {int(payload["job"]["data_seed"]) for payload, _ in payloads}
    dirty = [
        payload["job"]["job_id"]
        for payload, _ in payloads
        if payload["provenance"]["git_dirty"]
    ]
    failed_test = [
        payload["job"]["job_id"]
        for payload, _ in payloads
        if payload["test"].get("rollout_nrmse") is None
    ]
    report = {
        "publication_ready": observed == expected_seeds
        and not dirty
        and not failed_test,
        "observed_seeds": sorted(observed),
        "missing_seeds": sorted(expected_seeds - observed),
        "dirty_provenance_jobs": dirty,
        "failed_test_rollout_jobs": failed_test,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    with Path(__file__).with_name("config.toml").open("rb") as stream:
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
    payloads = collect_payloads(args.raw_directory, experiment_name)
    if not payloads:
        raise SystemExit(
            "No Figure 3 production results with trajectory artifacts found."
        )
    aggregate = args.output_directory / "aggregated"
    figures = args.output_directory / "figures"
    write_csv(aggregate / "figure_3_runs.csv", payloads)
    write_coefficient_csv(aggregate / "figure_3_coefficients.csv", payloads, config)
    report = write_quality(aggregate / "figure_3_quality.json", payloads, config)
    median_validation = float(
        np.median(
            [pair[0]["selected"]["validation"]["rollout_nrmse"] for pair in payloads]
        )
    )
    representative = min(
        payloads,
        key=lambda pair: abs(
            pair[0]["selected"]["validation"]["rollout_nrmse"] - median_validation
        ),
    )
    plot_joint_result(
        figures / "figure_3_joint_goodwin",
        representative[0],
        representative[1],
        payloads,
        len(config["experiment"]["seeds"]),
    )
    plot_full_library(figures / "figure_3_full_library", payloads, config)
    plot_selection_frequency(
        figures / "figure_3_selection_frequency", payloads, config
    )
    print(f"processed {len(payloads)} production results")
    print(f"publication ready: {report['publication_ready']}")
    print(f"wrote {figures / 'figure_3_joint_goodwin.png'}")


if __name__ == "__main__":
    main()
