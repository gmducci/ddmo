"""Aggregate Figure 3 runs and plot the joint-optimization demonstration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tomllib
from matplotlib.colors import TwoSlopeNorm

from benchmarks.common.goodwin import true_coefficient_matrix
from benchmarks.common.style import publication_style

plt.switch_backend("Agg")
STATE_NAMES = ["x", "y", "z"]
STATE_COLORS = ["#3B4CC0", "#00876C", "#B40426"]
PARETO_SAMPLE_SEED = 20260813


def _canonical_feature(name: str) -> str:
    if name.startswith("1/(1+"):
        variable = name.removeprefix("1/(1+").split("^")[0]
        return f"Hill({variable})"
    if name.startswith("exp("):
        variable = name.rsplit("*", 1)[-1].removesuffix(")")
        return f"exp({variable})"
    return name


def _library_feature_label(name: str) -> str:
    """Render polynomial library terms with math-text typography."""
    rendered_factors = []
    for factor in name.split("*"):
        variable, separator, exponent = factor.partition("^")
        if variable not in STATE_NAMES or (separator and not exponent.isdigit()):
            return name
        rendered_factors.append(
            variable if not separator else rf"{variable}^{{{exponent}}}"
        )
    return "$" + r"\,".join(rendered_factors) + "$"


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
        "optimizer_seed": payload["job"]["optimizer_seed"],
        "ic_block": payload["job"].get("ic_block"),
        "training_trajectories": payload["job"].get("training_trajectories", 3),
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


def sample_hill_active_pareto_solutions(
    payloads: list[tuple[dict[str, Any], Path]],
    config: dict[str, Any],
    random_seed: int = PARETO_SAMPLE_SEED,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select one reproducible Hill-active member from each eligible front."""
    generator = np.random.default_rng(random_seed)
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    samples = []
    excluded_seeds = []
    for payload, _ in payloads:
        pareto = payload["pareto"]
        if not pareto:
            raise ValueError(f"Empty Pareto front for {payload['job']['job_id']}")
        eligible = []
        for solution in pareto:
            hill_active = any(
                _canonical_feature(feature).startswith("Hill(")
                and np.any(np.abs(np.asarray(row, dtype=float)) > tolerance)
                for feature, row in zip(
                    solution["feature_names"], solution["coefficients"]
                )
            )
            if hill_active:
                eligible.append(solution)
        if not eligible:
            excluded_seeds.append(int(payload["job"]["data_seed"]))
            continue
        solution = eligible[int(generator.integers(len(eligible)))]
        samples.append(
            {
                "job_id": payload["job"]["job_id"],
                "data_seed": int(payload["job"]["data_seed"]),
                "pareto_size": len(pareto),
                "hill_active_pool_size": len(eligible),
                "pareto_solution_index": int(solution["index"]),
                "variables": {
                    name: float(value) for name, value in solution["variables"].items()
                },
            }
        )
    if not samples:
        raise ValueError("No Figure 3 Pareto front contains a Hill-active solution")
    return samples, excluded_seeds


def write_pareto_sample_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    """Record the exact random Pareto members used in the violin plot."""
    rows = []
    for sample in samples:
        rows.append(
            {
                "sampling_seed": PARETO_SAMPLE_SEED,
                "job_id": sample["job_id"],
                "data_seed": sample["data_seed"],
                "pareto_size": sample["pareto_size"],
                "hill_active_pool_size": sample["hill_active_pool_size"],
                "pareto_solution_index": sample["pareto_solution_index"],
                **sample["variables"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_optimization_variables(
    output_stem: Path,
    samples: list[dict[str, Any]],
    total_fronts: int,
    excluded_seeds: list[int],
    config: dict[str, Any],
) -> None:
    """Plot variables from sampled Hill-active Pareto members."""
    variable_specs = [
        ("gamma_1", r"Hill exponent $\gamma_1$", "linear"),
        ("gamma_2", r"Exponential rate $\gamma_2$", "linear"),
        ("degree", "Weak polynomial degree", "linear"),
        ("window_length", "Weak window length", "linear"),
        ("stride", "Weak stride", "linear"),
        ("threshold", "STLSQ threshold", "log"),
        ("alpha", r"STLSQ ridge $\alpha$", "log"),
    ]
    bound_sections = {
        "gamma_1": "library",
        "gamma_2": "library",
        "degree": "formulation",
        "window_length": "formulation",
        "stride": "formulation",
        "threshold": "solver",
        "alpha": "solver",
    }
    seeds = [int(sample["data_seed"]) for sample in samples]
    point_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.4, len(samples)))
    with publication_style():
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.8))
        for axis, (name, label, scale) in zip(axes.flat, variable_specs):
            values = np.asarray(
                [sample["variables"][name] for sample in samples], dtype=float
            )
            parts = axis.violinplot(
                values,
                positions=[0.0],
                widths=0.72,
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            for body in parts["bodies"]:
                body.set_facecolor("#5B8DB8")
                body.set_edgecolor("#264B6A")
                body.set_alpha(0.42)
            for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
                parts[part_name].set_color("#264B6A")
                parts[part_name].set_linewidth(1.0)
            offsets = np.linspace(-0.11, 0.11, len(values))
            for offset, value, color in zip(offsets, values, point_colors):
                axis.scatter(
                    offset,
                    value,
                    s=22,
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )
            section = bound_sections[name]
            lower, upper = map(float, config["bounds"][section][name])
            axis.set_ylim(lower, upper)
            axis.set_yscale(scale)
            axis.set_xlim(-0.52, 0.52)
            axis.set_xticks([])
            axis.set_title(label, fontweight="bold")
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            if name == "gamma_1":
                axis.axhline(
                    float(config["truth"]["hill_exponent"]),
                    color="#222222",
                    linestyle="--",
                    linewidth=0.9,
                    label="Ground truth",
                )
        unused_axis = axes.flat[-1]
        unused_axis.axis("off")
        seed_handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=f"seed {seed}",
            )
            for seed, color in zip(seeds, point_colors)
        ]
        unused_axis.legend(
            handles=seed_handles,
            loc="center",
            title=f"Hill-active fronts ({len(samples)}/{total_fronts})",
            ncol=1,
        )
        fig.suptitle(
            "Figure 3 optimization variables — Hill-active Pareto samples",
            fontsize=10,
        )
        fig.text(
            0.5,
            0.012,
            f"Fixed sampling seed: {PARETO_SAMPLE_SEED}; "
            + f"excluded fronts without an active Hill term: {excluded_seeds}",
            ha="center",
            fontsize=7.5,
        )
        fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.94), w_pad=1.25, h_pad=1.15)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"))
        fig.savefig(output_stem.with_suffix(".pdf"))
        plt.close(fig)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a 95% Wilson score interval for a binomial proportion."""
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return center - radius, center + radius


def plot_training_size_study(
    output_stem: Path,
    payloads: list[tuple[dict[str, Any], Path]],
) -> None:
    """Summarize the paired LHS study across stacked training-set sizes."""
    sizes = sorted(
        {int(payload["job"]["training_trajectories"]) for payload, _ in payloads}
    )
    grouped = {
        size: [
            payload
            for payload, _ in payloads
            if int(payload["job"]["training_trajectories"]) == size
        ]
        for size in sizes
    }
    point_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, 10))
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
        rate_axis, error_axis, gamma_axis, fp_axis = axes.flat
        rate_specs = [
            ("Hill selected", lambda payload: _family_selected(payload, "Hill(")),
            (
                "Exact support",
                lambda payload: bool(
                    payload["selected"]["coefficient_metrics"]["support_exact"]
                ),
            ),
            (
                "Exponential selected",
                lambda payload: _family_selected(payload, "exp("),
            ),
        ]
        rate_colors = [STATE_COLORS[2], STATE_COLORS[1], STATE_COLORS[0]]
        for offset, ((label, predicate), color) in enumerate(
            zip(rate_specs, rate_colors)
        ):
            x_values = np.asarray(sizes, dtype=float) + (offset - 1) * 0.07
            rates = []
            lower_errors = []
            upper_errors = []
            for size in sizes:
                values = grouped[size]
                successes = sum(predicate(payload) for payload in values)
                rate = successes / len(values)
                lower, upper = _wilson_interval(successes, len(values))
                rates.append(rate)
                lower_errors.append(rate - lower)
                upper_errors.append(upper - rate)
            rate_axis.errorbar(
                x_values,
                rates,
                yerr=np.asarray([lower_errors, upper_errors]),
                marker="o",
                color=color,
                capsize=2.5,
                label=label,
            )
        rate_axis.set(
            xlabel="Training trajectories",
            ylabel="Selection probability",
            xticks=sizes,
            ylim=(-0.04, 1.04),
        )
        rate_axis.legend(frameon=False, fontsize=7)

        metric_specs = [
            (
                error_axis,
                "Held-out test NRMSE",
                lambda payload: float(payload["test"]["rollout_nrmse"]),
            ),
            (
                fp_axis,
                "False-positive coefficients",
                lambda payload: float(
                    payload["selected"]["coefficient_metrics"][
                        "false_positive_coefficients"
                    ]
                ),
            ),
        ]
        for axis, ylabel, accessor in metric_specs:
            values_by_size = [
                np.asarray([accessor(payload) for payload in grouped[size]])
                for size in sizes
            ]
            axis.boxplot(values_by_size, positions=sizes, widths=0.45, showfliers=False)
            for size, values in zip(sizes, values_by_size):
                blocks = [int(payload["job"]["ic_block"]) for payload in grouped[size]]
                jitter = np.linspace(-0.11, 0.11, len(values))
                for dx, value, block in zip(jitter, values, blocks):
                    axis.scatter(
                        size + dx,
                        value,
                        color=point_colors[block - 1],
                        s=18,
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=3,
                    )
            axis.set(xlabel="Training trajectories", ylabel=ylabel, xticks=sizes)

        for size in sizes:
            active = [
                payload
                for payload in grouped[size]
                if _family_selected(payload, "Hill(")
            ]
            for dx, payload in zip(
                np.linspace(-0.11, 0.11, len(active)) if active else [], active
            ):
                block = int(payload["job"]["ic_block"])
                error = abs(float(payload["selected"]["variables"]["gamma_1"]) - 9.5)
                gamma_axis.scatter(
                    size + dx,
                    error,
                    color=point_colors[block - 1],
                    s=20,
                    edgecolor="white",
                    linewidth=0.35,
                )
        gamma_axis.set(
            xlabel="Training trajectories",
            ylabel=r"Conditional $|\hat{\gamma}_1-9.5|$",
            xticks=sizes,
        )
        for axis in axes.flat:
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        fig.suptitle("Figure 3 paired Latin-hypercube training-size study", fontsize=10)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=1.3, h_pad=1.2)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"))
        fig.savefig(output_stem.with_suffix(".pdf"))
        plt.close(fig)


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
    state = "PREVIEW" if payload["provenance"]["git_dirty"] else "FINAL"
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
            + rf"Hill selected {len(hill_payloads)}/{expected_jobs}; $\gamma_1$ median "
            + rf"{median_gamma_1:.4g}, range "
            + rf"[{np.min(recovered_gamma_1):.4g}, {np.max(recovered_gamma_1):.4g}]"
            + "\n"
            + rf"Exponential selected {len(exponential_payloads)}/{expected_jobs}; $\gamma_2$ range "
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


def plot_exponential_substitution(
    output_stem: Path,
    payload: dict[str, Any],
    artifact_path: Path,
    config: dict[str, Any],
) -> None:
    """Plot a held-out rollout for a model that substitutes an exponential term."""
    selected = payload["selected"]
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    coefficients = np.asarray(selected["coefficients"], dtype=float)
    exponential_terms = []
    for feature_index, feature in enumerate(selected["feature_names"]):
        canonical = _canonical_feature(feature)
        if not canonical.startswith("exp("):
            continue
        for equation_index, coefficient in enumerate(coefficients[feature_index]):
            if abs(coefficient) > tolerance:
                exponential_terms.append(
                    (canonical, equation_index, float(coefficient))
                )
    if not exponential_terms:
        raise ValueError("Selected Figure 3 model has no active exponential term")

    feature, equation_index, coefficient = exponential_terms[0]
    gamma_2 = float(selected["variables"]["gamma_2"])
    gamma_truth = float(config["truth"]["hill_exponent"])
    arrays = np.load(artifact_path)
    time = arrays["time"]
    clean = arrays["test_clean"][0]
    noisy = arrays["test_noisy"][0]
    prediction = arrays["test_prediction"][0]
    observed_z = np.concatenate(
        [
            arrays["train_clean"][:, :, 2].ravel(),
            arrays["validation_clean"][:, :, 2].ravel(),
            arrays["test_clean"][:, :, 2].ravel(),
        ]
    )
    plot_min = max(0.0, float(np.min(observed_z)))
    plot_max = min(2.0, float(np.max(observed_z)))
    library_input = np.linspace(plot_min, plot_max, 500)
    learned_source = coefficient * np.exp(-gamma_2 * library_input)
    truth_source = 1.0 / (1.0 + library_input**gamma_truth)
    training_min_z = float(np.min(arrays["train_clean"][:, :, 2]))
    learned_at_training_min = coefficient * np.exp(-gamma_2 * training_min_z)
    truth_at_training_min = 1.0 / (1.0 + training_min_z**gamma_truth)
    test_nrmse = float(payload["test"]["rollout_nrmse"])

    with publication_style():
        fig = plt.figure(figsize=(7.2, 5.2))
        grid = fig.add_gridspec(
            3, 2, width_ratios=[2.1, 1.55], hspace=0.28, wspace=0.34
        )
        handles = labels = None
        for state_index, (state_name, color) in enumerate(
            zip(STATE_NAMES, STATE_COLORS)
        ):
            axis = fig.add_subplot(grid[state_index, 0])
            axis.plot(
                time,
                clean[:, state_index],
                color="#222222",
                label="Clean reference",
            )
            axis.scatter(
                time[::10],
                noisy[::10, state_index],
                color=color,
                s=8,
                alpha=0.3,
                label="Noisy observations",
            )
            axis.plot(
                time,
                prediction[:, state_index],
                color=color,
                linestyle="--",
                label="Identified rollout",
            )
            axis.set_ylabel(state_name)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            if state_index < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Time")
            if state_index == 0:
                handles, labels = axis.get_legend_handles_labels()

        source_axis = fig.add_subplot(grid[0:2, 1])
        source_axis.plot(
            library_input,
            truth_source,
            color="#222222",
            linewidth=1.6,
            label=rf"Truth: $1/(1+z^{{{gamma_truth:.3g}}})$",
        )
        source_axis.plot(
            library_input,
            learned_source,
            color=STATE_COLORS[2],
            linewidth=1.6,
            linestyle="--",
            label=rf"Selected: ${coefficient:.3g}e^{{-{gamma_2:.3g}z}}$",
        )
        source_axis.axhline(0.0, color="#BDBDBD", linewidth=0.7, zorder=0)
        source_axis.set(
            xlabel="z (observed switching region)",
            ylabel=r"Contribution to $\dot{x}$",
            xlim=(plot_min, plot_max),
        )
        source_axis.axvline(
            training_min_z,
            color="#808080",
            linewidth=0.8,
            linestyle=":",
            label="Minimum training z",
        )
        source_axis.set_title("Source terms near switching region", loc="left")
        source_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        source_axis.legend(frameon=False, fontsize=7)

        detail_axis = fig.add_subplot(grid[2, 1])
        detail_axis.axis("off")
        detail_axis.text(
            0.0,
            0.82,
            rf"Selected term: {feature} in $\dot{{{STATE_NAMES[equation_index]}}}$"
            + "\n"
            + rf"coefficient={coefficient:.6g}, $\gamma_2$={gamma_2:.6g}"
            + "\n"
            + rf"at min training z={training_min_z:.3g}: selected="
            + rf"{learned_at_training_min:.3g}, truth={truth_at_training_min:.3g}"
            + "\n"
            + rf"held-out test NRMSE={test_nrmse:.4g}",
            ha="left",
            va="top",
            fontsize=7.5,
            linespacing=1.5,
        )
        fig.suptitle(
            f"Exponential-substitution rollout — seed {payload['job']['data_seed']}",
            fontsize=10,
            y=0.995,
        )
        if handles is not None and labels is not None:
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


def plot_library_identifiability(
    output_stem: Path,
    payload: dict[str, Any],
    artifact_path: Path,
    config: dict[str, Any],
) -> None:
    """Compare fitted Hill and exponential source contributions on observed z."""
    selected = payload["selected"]
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    coefficients = np.asarray(selected["coefficients"], dtype=float)
    hill_index = next(
        (
            index
            for index, feature in enumerate(selected["feature_names"])
            if _canonical_feature(feature) == "Hill(z)"
            and abs(coefficients[index, 0]) > tolerance
        ),
        None,
    )
    exponential_index = next(
        (
            index
            for index, feature in enumerate(selected["feature_names"])
            if _canonical_feature(feature) == "exp(z)"
            and abs(coefficients[index, 0]) > tolerance
        ),
        None,
    )
    if hill_index is None or exponential_index is None:
        raise ValueError(
            "Selected model must contain active z-Hill and z-exponential terms "
            "in dx/dt"
        )

    gamma_1 = float(selected["variables"]["gamma_1"])
    gamma_2 = float(selected["variables"]["gamma_2"])
    gamma_truth = float(config["truth"]["hill_exponent"])
    hill_coefficient = float(coefficients[hill_index, 0])
    exponential_coefficient = float(coefficients[exponential_index, 0])
    arrays = np.load(artifact_path)
    training_z = arrays["train_clean"][:, :, 2].ravel()

    plot_min = max(0.0, float(np.min(training_z)) * 0.97)
    plot_max = float(np.quantile(training_z, 0.85))
    library_input = np.linspace(plot_min, plot_max, 600)
    truth = 1.0 / (1.0 + library_input**gamma_truth)
    fitted_hill = hill_coefficient / (1.0 + library_input**gamma_1)
    fitted_exponential = exponential_coefficient * np.exp(-gamma_2 * library_input)
    fitted_sum = fitted_hill + fitted_exponential

    observed_hill = 1.0 / (1.0 + training_z**gamma_1)
    observed_exponential = np.exp(-gamma_2 * training_z)
    column_correlation = float(np.corrcoef(observed_hill, observed_exponential)[0, 1])
    observed_truth = 1.0 / (1.0 + training_z**gamma_truth)
    observed_sum = (
        hill_coefficient * observed_hill
        + exponential_coefficient * observed_exponential
    )
    contribution_rmse = float(np.sqrt(np.mean((observed_sum - observed_truth) ** 2)))

    with publication_style():
        fig, (axis, density_axis) = plt.subplots(
            2,
            1,
            figsize=(6.5, 4.6),
            sharex=True,
            gridspec_kw={"height_ratios": [4.0, 1.0], "hspace": 0.08},
        )
        axis.plot(
            library_input,
            truth,
            color="#222222",
            linewidth=1.8,
            label=rf"Ground truth Hill ($\gamma={gamma_truth:.3g}$)",
        )
        axis.plot(
            library_input,
            fitted_hill,
            color=STATE_COLORS[0],
            linewidth=1.4,
            linestyle="--",
            label=(
                rf"Fitted Hill ($c={hill_coefficient:.3g}$, "
                rf"$\gamma_1={gamma_1:.3g}$)"
            ),
        )
        axis.plot(
            library_input,
            fitted_exponential,
            color=STATE_COLORS[2],
            linewidth=1.4,
            linestyle=":",
            label=(
                rf"Fitted exponential ($c={exponential_coefficient:.3g}$, "
                rf"$\gamma_2={gamma_2:.3g}$)"
            ),
        )
        axis.plot(
            library_input,
            fitted_sum,
            color=STATE_COLORS[1],
            linewidth=1.8,
            label="Fitted Hill + exponential",
        )
        axis.set_ylabel(r"Contribution to $\dot{x}$")
        axis.set_ylim(bottom=min(0.0, float(np.min(fitted_sum))))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.legend(frameon=False, fontsize=7.5)
        axis.set_title(
            rf"Representative mixed model ($N_{{train}}="
            + f"{payload['job']['training_trajectories']}$, "
            + f"block {payload['job']['ic_block']})",
            loc="left",
            fontsize=8,
        )
        axis.text(
            0.98,
            0.96,
            rf"library-column $r={column_correlation:.3f}$"
            + "\n"
            + rf"combined contribution RMSE={contribution_rmse:.3g}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )

        density_axis.hist(
            training_z,
            bins=45,
            range=(plot_min, plot_max),
            density=True,
            color="#BDBDBD",
            edgecolor="none",
        )
        density_axis.set(
            xlabel=r"Observed training state $z$ (through the 85th percentile)",
            ylabel="Density",
            xlim=(plot_min, plot_max),
        )
        density_axis.grid(False)
        fig.suptitle(
            "Hill--exponential identifiability on the observed domain",
            fontsize=10,
            y=0.995,
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
    labels = [_library_feature_label(name) for name in names]
    estimates = []
    for payload, _ in payloads:
        selected = payload["selected"]
        current_names = [_canonical_feature(name) for name in selected["feature_names"]]
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
            axis.set_yticks(range(len(names)), labels)
            axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.4)
            axis.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(
            image, ax=axes, orientation="horizontal", fraction=0.045, pad=0.10
        )
        colorbar.set_label("Coefficient")
        fig.suptitle(
            "Figure 3 complete polynomial + Hill + exponential library", fontsize=10
        )
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
    labels = [_library_feature_label(name) for name in names]
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    estimates = []
    for payload, _ in payloads:
        selected = payload["selected"]
        current_names = [_canonical_feature(name) for name in selected["feature_names"]]
        matrix = np.asarray(selected["coefficients"], dtype=float)
        order = [current_names.index(name) for name in names]
        estimates.append(matrix[order])
    estimates_array = np.asarray(estimates, dtype=float)
    supports = np.abs(estimates_array) > tolerance
    frequency = np.mean(supports, axis=0)
    conditional_median = np.full(frequency.shape, np.nan, dtype=float)
    for feature_index, equation_index in np.ndindex(frequency.shape):
        selected_coefficients = estimates_array[
            supports[:, feature_index, equation_index],
            feature_index,
            equation_index,
        ]
        if selected_coefficients.size:
            conditional_median[feature_index, equation_index] = np.median(
                selected_coefficients
            )
    truth = true_coefficient_matrix(reference_names, config)
    coefficient_limit = float(np.max(np.abs(truth)))
    coefficients_clipped = bool(
        np.nanmax(np.abs(conditional_median)) > coefficient_limit
    )
    coefficient_norm = TwoSlopeNorm(
        vmin=-coefficient_limit, vcenter=0.0, vmax=coefficient_limit
    )
    with publication_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.0), sharey=True)
        frequency_axis, coefficient_axis, truth_axis = axes
        frequency_image = frequency_axis.imshow(
            frequency,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="nearest",
        )
        coefficient_image = coefficient_axis.imshow(
            conditional_median,
            cmap="RdBu_r",
            norm=coefficient_norm,
            aspect="auto",
            interpolation="nearest",
        )
        truth_axis.imshow(
            truth,
            cmap="RdBu_r",
            norm=coefficient_norm,
            aspect="auto",
            interpolation="nearest",
        )
        equation_labels = [r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"]
        for axis in axes:
            axis.set_xticks(range(3), equation_labels)
            axis.set_yticks(range(len(names)), labels)
            axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.4)
            axis.tick_params(which="minor", bottom=False, left=False)
        for row, column in zip(*np.where(frequency > 0)):
            frequency_axis.text(
                column,
                row,
                f"{100 * frequency[row, column]:.0f}%",
                ha="center",
                va="center",
                color="white" if frequency[row, column] < 0.65 else "black",
                fontsize=7,
            )
        for axis, matrix in (
            (coefficient_axis, conditional_median),
            (truth_axis, truth),
        ):
            for row, column in np.ndindex(matrix.shape):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.2g}",
                        ha="center",
                        va="center",
                        color="white"
                        if abs(value) > 0.55 * coefficient_limit
                        else "black",
                        fontsize=6.5,
                    )
        frequency_axis.set_title("Selection frequency", fontweight="bold")
        coefficient_axis.set_title(
            "Median coefficient\nwhen selected", fontweight="bold"
        )
        truth_axis.set_title("Ground-truth\ncoefficient", fontweight="bold")
        fig.subplots_adjust(
            left=0.20, right=0.985, top=0.92, bottom=0.22, wspace=0.12
        )

        frequency_bounds = frequency_axis.get_position()
        coefficient_bounds = coefficient_axis.get_position()
        truth_bounds = truth_axis.get_position()
        colorbar_width = frequency_bounds.width
        colorbar_height = 0.022
        colorbar_bottom = 0.085
        frequency_colorbar_axis = fig.add_axes(
            (
                frequency_bounds.x0,
                colorbar_bottom,
                colorbar_width,
                colorbar_height,
            )
        )
        coefficient_group_center = (
            coefficient_bounds.x0 + truth_bounds.x1
        ) / 2.0
        coefficient_colorbar_axis = fig.add_axes(
            (
                coefficient_group_center - colorbar_width / 2.0,
                colorbar_bottom,
                colorbar_width,
                colorbar_height,
            )
        )
        frequency_colorbar = fig.colorbar(
            frequency_image,
            cax=frequency_colorbar_axis,
            orientation="horizontal",
        )
        frequency_colorbar.set_label("Fraction of seeds")
        coefficient_colorbar = fig.colorbar(
            coefficient_image,
            cax=coefficient_colorbar_axis,
            orientation="horizontal",
        )
        coefficient_colorbar.set_label(
            "Coefficient"
            + (
                " (colors clipped to ground-truth range)"
                if coefficients_clipped
                else ""
            )
        )
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
    pareto_samples, excluded_pareto_seeds = sample_hill_active_pareto_solutions(
        payloads, config
    )
    write_pareto_sample_csv(
        aggregate / "figure_3_random_pareto_variables.csv", pareto_samples
    )
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
    plot_selection_frequency(figures / "figure_3_selection_frequency", payloads, config)
    plot_optimization_variables(
        figures / "figure_3_optimization_variables",
        pareto_samples,
        len(payloads),
        excluded_pareto_seeds,
        config,
    )
    if all("training_trajectories" in payload["job"] for payload, _ in payloads):
        plot_training_size_study(figures / "figure_3_training_size_study", payloads)
    exponential_payloads = [
        pair for pair in payloads if _family_selected(pair[0], "exp(")
    ]
    if exponential_payloads:
        representative_exponential = min(
            exponential_payloads,
            key=lambda pair: pair[0]["selected"]["validation"]["rollout_nrmse"],
        )
        plot_exponential_substitution(
            figures / "figure_3_exponential_substitution_dynamics",
            representative_exponential[0],
            representative_exponential[1],
            config,
        )
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    mixed_payloads = []
    for pair in payloads:
        selected = pair[0]["selected"]
        coefficients = np.asarray(selected["coefficients"], dtype=float)
        has_hill = any(
            _canonical_feature(feature) == "Hill(z)"
            and abs(coefficients[index, 0]) > tolerance
            for index, feature in enumerate(selected["feature_names"])
        )
        has_exponential = any(
            _canonical_feature(feature) == "exp(z)"
            and abs(coefficients[index, 0]) > tolerance
            for index, feature in enumerate(selected["feature_names"])
        )
        if has_hill and has_exponential:
            mixed_payloads.append(pair)
    if mixed_payloads:
        representative_mixed = min(
            mixed_payloads,
            key=lambda pair: pair[0]["selected"]["validation"]["rollout_nrmse"],
        )
        plot_library_identifiability(
            figures / "figure_3_library_identifiability",
            representative_mixed[0],
            representative_mixed[1],
            config,
        )
    print(f"processed {len(payloads)} production results")
    print(f"publication ready: {report['publication_ready']}")
    print(f"wrote {figures / 'figure_3_joint_goodwin.png'}")


if __name__ == "__main__":
    main()
