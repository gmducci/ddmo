"""Supplementary comparison of Goodwin equation mismatches at true parameters.

This diagnostic uses saved held-out trajectories and production-selected weak
windows. It does not refit a model or use the fitted coefficients.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from benchmarks.common.data import trajectory_data
from benchmarks.common.goodwin import _savgol_states
from benchmarks.common.plan_runs import build_jobs
from benchmarks.common.style import publication_style
from ddmo.formulation.weak import WeakFormulation, _sliding_integral
from ddmo.utils.data_constructor import DataConstructor

plt.switch_backend("Agg")

FIGURE_NAME = "figure_s6_equation_mismatch"
METHODS = ("strong_sg", "strong_fd", "weak_raw", "weak_sg")
LABELS = {
    "strong_sg": "Strong + SG",
    "strong_fd": "Strong FD",
    "weak_raw": "Weak, optimized",
    "weak_sg": "Weak, optimized + SG",
}
COLORS = {
    "strong_sg": "#3B4CC0",
    "strong_fd": "#6B7280",
    "weak_raw": "#B40426",
    "weak_sg": "#E07A1F",
}
MARKERS = {"strong_sg": "s", "strong_fd": "*", "weak_raw": "o", "weak_sg": "D"}


def _goodwin_rhs(states: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Evaluate the configured true vector field at each observed state."""
    truth = config["truth"]
    x, y, z = np.moveaxis(states, -1, 0)
    exponent = float(truth["hill_exponent"])
    k1 = float(truth.get("k1", 1.0))
    k2 = float(truth.get("k2", 0.2))
    k3 = float(truth.get("k3", 1.0))
    k4 = float(truth.get("k4", 0.15))
    k5 = float(truth.get("k5", 1.0))
    k6 = float(truth.get("k6", 0.1))
    K1 = float(truth.get("K1", 1.0))
    return np.stack(
        (
            k1 * K1**exponent / (K1**exponent + z**exponent) - k2 * x,
            k3 * x - k4 * y,
            k5 * y - k6 * z,
        ),
        axis=-1,
    )


def _weak_mismatch(
    states: np.ndarray,
    time: np.ndarray,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return B - A Xi_true using the production weak quadrature weights."""
    data = DataConstructor(
        x_train=states,
        time_array=time,
        x_0=[row[0] for row in states],
        x_raw=states,
    )
    formulation = WeakFormulation(
        data_instance=data,
        class_attributes={
            "test_functions_class": settings["family"],
            "degree": int(settings["degree"]),
            "window_length": int(settings["window_length"]),
            "stride": int(settings["stride"]),
        },
    )
    lhs = formulation.get_lhs(data, ())
    rhs_fields = _goodwin_rhs(states, config)
    rhs = np.vstack(
        [
            np.column_stack(
                [
                    _sliding_integral(block[:, state], centers, formulation.w_rhs)
                    for state in range(block.shape[-1])
                ]
            )
            for block, centers in zip(rhs_fields, formulation.centers_by_trajectory)
        ]
    )
    centers = np.concatenate(formulation.centers_by_trajectory)
    return lhs - rhs, time[centers]


def _strong_mismatch(
    states: np.ndarray, time: np.ndarray, config: dict[str, Any], method: str
) -> np.ndarray:
    dt = float(time[1] - time[0])
    if method == "strong_fd":
        derivative = np.gradient(states, dt, axis=1, edge_order=2)
        right_states = states
    elif method == "strong_sg":
        settings = config["strong_derivative"]
        derivative = savgol_filter(
            states,
            window_length=int(settings["window_length"]),
            polyorder=int(settings["polynomial_order"]),
            deriv=1,
            delta=dt,
            axis=1,
        )
        right_states = _savgol_states(config, states)
    else:
        raise ValueError(f"Unknown strong method: {method}")
    return derivative - _goodwin_rhs(right_states, config)


def _score(mismatch: np.ndarray, derivative_scale: np.ndarray) -> float:
    """RMS after normalizing each state by its exact-derivative RMS."""
    return float(np.sqrt(np.mean((mismatch / derivative_scale) ** 2)))


def collect(
    config: dict[str, Any], raw_directory: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs = [
        job
        for job in build_jobs(config, Path("benchmarks/noise/config.toml"))
        if job["method"] in {"strong", "weak_optimized", "weak_optimized_sg"}
    ]
    grouped: dict[tuple[float, int], dict[str, str]] = {}
    for job in jobs:
        key = (float(job["noise_level"]), int(job["data_seed"]))
        grouped.setdefault(key, {})[job["method"]] = job["job_id"]

    records: list[dict[str, Any]] = []
    example: dict[str, Any] = {}
    weak_base = config.get("weak_initial", config.get("weak_fixed"))
    if weak_base is None:
        raise ValueError("No weak test-function family in the saved configuration")
    with trajectory_data("noise") as saved:
        for (noise, seed), by_method in sorted(grouped.items()):
            if set(by_method) != {"strong", "weak_optimized", "weak_optimized_sg"}:
                raise ValueError(f"Missing methods for noise={noise}, seed={seed}")
            arrays = saved[by_method["strong"]]
            for other_method in ("weak_optimized", "weak_optimized_sg"):
                other = saved[by_method[other_method]]
                for name in ("time", "test_clean", "test_noisy"):
                    if not np.array_equal(arrays[name], other[name]):
                        raise ValueError(f"Saved {name} differs across methods")

            time = arrays["time"]
            clean = arrays["test_clean"]
            noisy = arrays["test_noisy"]
            derivative_scale = np.sqrt(np.mean(_goodwin_rhs(clean, config) ** 2, axis=(0, 1)))
            if np.any(derivative_scale <= 0):
                raise ValueError("Expected nonzero derivative scale for every state")
            smoothed = _savgol_states(config, noisy)
            mismatch = {
                "strong_sg": _strong_mismatch(noisy, time, config, "strong_sg"),
                "strong_fd": _strong_mismatch(noisy, time, config, "strong_fd"),
            }
            weak_times = {}
            weak_settings = {}
            for method, job_method, observations in (
                ("weak_raw", "weak_optimized", noisy),
                ("weak_sg", "weak_optimized_sg", smoothed),
            ):
                payload = json.loads((raw_directory / f"{by_method[job_method]}.json").read_text())
                selected = payload["selected"]["variables"]
                settings = {
                    "family": weak_base["family"],
                    "degree": int(selected["degree"]),
                    "window_length": int(selected["window_length"]),
                    "stride": int(selected["stride"]),
                }
                mismatch[method], weak_times[method] = _weak_mismatch(
                    observations, time, config, settings
                )
                weak_settings[method] = settings
            for method in METHODS:
                record = {
                    "noise_level": noise,
                    "seed": seed,
                    "method": method,
                    "relative_equation_rms": _score(mismatch[method], derivative_scale),
                    "number_of_constraints": int(np.prod(mismatch[method].shape[:-1])),
                }
                if method in weak_settings:
                    record["weak_settings"] = weak_settings[method]
                records.append(record)
            if noise == 0 and seed == min(config["experiment"]["seeds"]):
                example = {
                    "time": time,
                    "weak_times": weak_times,
                    "mismatch": mismatch,
                    "derivative_scale": derivative_scale,
                    "job_id": by_method["strong"],
                }
    return records, example


def _plot(records: list[dict[str, Any]], example: dict[str, Any]) -> plt.Figure:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.2, 3.55), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    state_index = 0
    for method in METHODS:
        mismatch = example["mismatch"][method]
        if method.startswith("weak"):
            half = mismatch.shape[0] // 2
            time = example["weak_times"][method][:half]
            error = mismatch[:half, state_index]
        else:
            time = example["time"]
            error = mismatch[0, :, state_index]
        left.plot(
            time,
            np.abs(error),
            label=LABELS[method],
            color=COLORS[method],
            linewidth=1.15,
            alpha=0.9,
        )
    left.set_xlim(0, 15)
    left.set_yscale("log")
    left.set_xlabel("Time")
    left.set_ylabel(r"Absolute $x$-equation mismatch")
    left.set_title("a  Clean observations (0% noise)", loc="left", fontweight="bold")
    left.text(
        0.98,
        0.96,
        "Weak values shown at window centers\n(first windows begin at time 0)",
        transform=left.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )

    levels = sorted({record["noise_level"] for record in records})
    for method_index, method in enumerate(METHODS):
        medians = []
        for noise in levels:
            values = np.asarray(
                [
                    record["relative_equation_rms"]
                    for record in records
                    if record["noise_level"] == noise and record["method"] == method
                ]
            )
            medians.append(float(np.median(values)))
            offset = (method_index - 1.5) * 0.0022
            right.scatter(
                np.full(values.shape, noise + offset) * 100,
                values,
                color=COLORS[method],
                alpha=0.34,
                s=17,
                linewidth=0,
                zorder=2,
            )
        right.plot(
            np.asarray(levels) * 100,
            medians,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=5.2 if method != "strong_fd" else 7,
            linewidth=1.6,
            label=LABELS[method],
            zorder=3,
        )
    right.set_xticks(np.asarray(levels) * 100)
    right.set_xlabel("Observation noise (%)")
    right.set_ylabel("RMS equation mismatch / RMS true derivative")
    right.set_yscale("log")
    right.set_title("b  True-parameter equation mismatch", loc="left", fontweight="bold")

    for axis in (left, right):
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55)
        axis.set_axisbelow(True)
    handles, labels = right.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.19, top=0.76, wspace=0.33)
    return fig


def render(config: dict[str, Any], raw_directory: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records, example = collect(config, raw_directory)
    metadata = {
        "status": "diagnostic_at_true_parameters_with_saved_selected_weak_windows",
        "source": "saved held-out Goodwin trajectories in benchmarks/data/historical/noise_trajectories.npz",
        "normalization": (
            "RMS of each state mismatch divided by RMS of its exact clean derivative; "
            "then RMS across states"
        ),
        "figure_1_metric": (
            "Test rollout NRMSE is the mean across states of fitted-rollout RMSE "
            "divided by clean-state standard deviation; it is a different quantity"
        ),
        "weak_window": (
            "production-selected degree, length, and stride from each archived result; "
            "family is PolynomialWindow"
        ),
        "strong_sg": config["strong_derivative"],
        "strong_fd": "numpy.gradient(states, dt, axis=1, edge_order=2)",
        "weak_quadrature": "production WeakFormulation LHS and RHS weights",
        "example_job_id": example["job_id"],
        "records": records,
    }
    (output / f"{FIGURE_NAME}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with publication_style():
        fig = _plot(records, example)
        fig.savefig(
            output / f"{FIGURE_NAME}.pdf",
            metadata={
                "CreationDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
                "ModDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
            },
        )
        fig.savefig(output / f"{FIGURE_NAME}.png")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    from benchmarks.common.data import result_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path(".local/benchmark_plots"))
    args = parser.parse_args()
    with result_data("noise") as (saved_config, raw):
        render(saved_config, raw, args.output_directory)
