"""Historical supplementary diagnostics from saved production trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from benchmarks.common.goodwin import true_coefficient_matrix
from benchmarks.common.style import publication_style
from benchmarks.joint.postprocess import _canonical_feature
from benchmarks.noise.postprocess import (
    METHOD_COLORS as NOISE_METHOD_COLORS,
)
from benchmarks.noise.postprocess import (
    METHOD_LABELS as NOISE_METHOD_LABELS,
)
from benchmarks.noise.postprocess import (
    STATE_COLORS,
    STATE_NAMES,
)

plt.switch_backend("Agg")

DISPLAY_METHODS = ("strong", "weak_optimized", "weak_optimized_sg")
METHOD_COLORS = {**NOISE_METHOD_COLORS, "weak_joint_sg": NOISE_METHOD_COLORS["weak_optimized_sg"]}
METHOD_LABELS = {**NOISE_METHOD_LABELS, "weak_joint_sg": "Weak, optimized + SG"}

OBJECTIVE_LABELS = {
    "trajectory_mse": "Training trajectory MSE",
    "trajectory_rmse": "Training trajectory RMSE",
    "durbin_watson_deviation": r"$|\mathrm{DW}-2|$",
    "active_coefficients": "Active coefficients",
    "active_coefficient_count": "Active coefficients",
}


@dataclass(frozen=True)
class SupplementaryRun:
    payload: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def method(self) -> str:
        return str(self.payload["job"]["method"])

    @property
    def noise_level(self) -> float:
        return float(self.payload["job"]["noise_level"])

    @property
    def seed(self) -> int:
        return int(self.payload["job"]["data_seed"])


def _shared_test_data(
    method_runs: list[SupplementaryRun],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
    artifacts = [run.arrays for run in method_runs]
    reference = artifacts[0]
    for run, artifact in zip(method_runs[1:], artifacts[1:]):
        for name in ("time", "test_clean", "test_noisy"):
            if not np.array_equal(reference[name], artifact[name]):
                raise RuntimeError(
                    f"Test array {name!r} is not shared across methods for "
                    f"noise={run.noise_level}, seed={run.seed}"
                )
    return (
        reference["time"],
        reference["test_clean"],
        reference["test_noisy"],
        artifacts,
    )


def _feature_label(name: str) -> str:
    if name.startswith("1/(1+"):
        variable = name.removeprefix("1/(1+").split("^")[0]
        return rf"$\mathrm{{Hill}}({variable})$"
    rendered = []
    for factor in name.split("*"):
        variable, separator, exponent = factor.partition("^")
        if variable not in STATE_NAMES or (separator and not exponent.replace(".", "").isdigit()):
            return name
        rendered.append(
            variable if not separator else rf"{variable}^{{{exponent}}}"
        )
    return "$" + r"\,".join(rendered) + "$"


def _coefficient_text(value: float, tolerance: float) -> str:
    if abs(value) <= tolerance:
        return "—"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _padded_limits(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([value[np.isfinite(value)].ravel() for value in values])
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    padding = 0.06 * max(upper - lower, 1.0)
    return lower - padding, upper + padding


def _style_time_axis(axis: plt.Axes, time: np.ndarray, ylabel: str | None) -> None:
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_xlabel("Time")
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color("#A8A8A8")
        spine.set_linewidth(0.6)


def _plot_noisy_test(
    axis: plt.Axes,
    time: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    trajectory_index: int,
) -> None:
    step = max(1, len(time) // 350)
    for state_index, (state, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
        axis.plot(
            time,
            clean[trajectory_index, :, state_index],
            color=color,
            linewidth=0.85,
            alpha=0.9,
        )
        axis.scatter(
            time[::step],
            noisy[trajectory_index, ::step, state_index],
            color=color,
            s=4.0,
            alpha=0.28,
            edgecolors="none",
            rasterized=True,
        )
    axis.set_title(f"Noisy test data · trajectory {trajectory_index + 1}", pad=4)
    _style_time_axis(axis, time, "State value" if trajectory_index == 0 else None)


def _plot_reconstruction(
    axis: plt.Axes,
    time: np.ndarray,
    clean: np.ndarray,
    prediction: np.ndarray,
    trajectory_index: int,
) -> None:
    if not np.any(np.isfinite(prediction[trajectory_index])):
        axis.text(0.5, 0.5, "Rollout failed", transform=axis.transAxes, ha="center", va="center", fontsize=9)
    for state_index, color in enumerate(STATE_COLORS):
        axis.plot(
            time,
            clean[trajectory_index, :, state_index],
            color=color,
            linewidth=0.85,
        )
        axis.plot(
            time,
            prediction[trajectory_index, :, state_index],
            color=color,
            linewidth=1.0,
            linestyle=(0, (3.2, 2.0)),
        )
    axis.set_title(f"Test reconstruction · trajectory {trajectory_index + 1}", pad=4)
    _style_time_axis(axis, time, None)


def _plot_pareto(axis: plt.Axes, run: SupplementaryRun) -> None:
    pareto = list(run.payload["pareto"])
    objective_names = list(pareto[0]["objectives"])
    if len(objective_names) not in (2, 3):
        raise RuntimeError(f"Expected two or three Pareto objectives; found {objective_names}")
    first = next((name for name in objective_names if name in
                  ("trajectory_rmse", "trajectory_mse")), None)
    second = next((name for name in objective_names if name in
                   ("active_coefficients", "active_coefficient_count")), None)
    third = "durbin_watson_deviation" if len(objective_names) == 3 else None
    if first is None or second is None or (third and third not in objective_names):
        raise RuntimeError(f"Unsupported Pareto objectives: {objective_names}")
    x_values = np.asarray([item["objectives"][first] for item in pareto], dtype=float)
    y_values = np.asarray([item["objectives"][second] for item in pareto], dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    colors = {}
    if third:
        dw = np.asarray([item["objectives"][third] for item in pareto], dtype=float)
        finite &= np.isfinite(dw)
        colors = {"c": dw[finite], "cmap": "cividis", "vmin": 0.0, "vmax": 2.0}
    else:
        colors = {"color": "#8A8A8A"}
    points = axis.scatter(
        x_values[finite], y_values[finite], s=18, alpha=0.8,
        edgecolors="none", rasterized=True, **colors,
    )
    if third:
        colorbar = axis.figure.colorbar(points, ax=axis, pad=0.03, fraction=0.045)
        colorbar.set_label(OBJECTIVE_LABELS[third], fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
    selected_index = int(run.payload["selected_index"])
    selected = next(item for item in pareto if int(item["index"]) == selected_index)
    axis.scatter(
        selected["objectives"][first],
        selected["objectives"][second],
        marker="*",
        s=100,
        color=METHOD_COLORS[run.method],
        edgecolor="#222222",
        linewidth=0.6,
        zorder=4,
        label="Validation-selected model",
    )
    positive_x = x_values[finite & (x_values > 0)]
    if positive_x.size and positive_x.size == np.count_nonzero(finite) and (
        float(np.max(positive_x)) / float(np.min(positive_x)) > 20.0
    ):
        axis.set_xscale("log")
    axis.set_xlabel(OBJECTIVE_LABELS.get(first, first.replace("_", " ")))
    axis.set_ylabel(OBJECTIVE_LABELS.get(second, second.replace("_", " ")))
    axis.set_title(f"Pareto projection · {np.count_nonzero(finite)} models" if third
                   else f"Pareto front · {np.count_nonzero(finite)} models", pad=4)
    if third:
        selected_dw = float(selected["objectives"][third])
        axis.text(0.98, 0.98, rf"Selected $|\mathrm{{DW}}-2|$={selected_dw:.3g}",
                  transform=axis.transAxes, ha="right", va="top", fontsize=6.5)
    axis.grid(color="#E0E0E0", linewidth=0.5)
    axis.legend(loc="best", fontsize=6.2, frameon=False)


def _plot_coefficient_table(
    axis: plt.Axes,
    run: SupplementaryRun,
    config: dict[str, Any],
) -> None:
    selected = run.payload["selected"]
    names = list(selected["feature_names"])
    coefficients = np.asarray(selected["coefficients"], dtype=float)
    truth = true_coefficient_matrix(names, config)
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    if coefficients.shape != truth.shape:
        raise RuntimeError(
            f"Coefficient shape mismatch for {run.payload['job']['job_id']}: "
            f"estimate={coefficients.shape}, truth={truth.shape}"
        )
    cell_text = [
        [_coefficient_text(value, tolerance) for value in row]
        for row in coefficients
    ]
    table = axis.table(
        cellText=cell_text,
        rowLabels=[_feature_label(name) for name in names],
        colLabels=[r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"],
        cellLoc="center",
        rowLoc="right",
        colLoc="center",
        bbox=(0.0, 0.0, 1.0, 0.93),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.7)
    table.scale(1.0, 1.08)
    for column in range(coefficients.shape[1]):
        header = table[(0, column)]
        header.set_facecolor("#E8E8E8")
        header.set_text_props(fontweight="bold")
    for row in range(coefficients.shape[0]):
        row_label = table[(row + 1, -1)]
        row_label.set_facecolor("#F3F3F3")
        for column in range(coefficients.shape[1]):
            estimate_active = abs(coefficients[row, column]) > tolerance
            truth_active = abs(truth[row, column]) > tolerance
            cell = table[(row + 1, column)]
            if estimate_active and truth_active:
                cell.set_facecolor("#DCEEDB")
            elif estimate_active:
                cell.set_facecolor("#FCE2C4")
            elif truth_active:
                cell.set_facecolor("#F4CACA")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#D0D0D0")
            cell.set_linewidth(0.35)
    metrics = selected["coefficient_metrics"]
    axis.set_title(
        "Selected coefficient table"
        + f" · FP={metrics['false_positive_coefficients']}"
        + f", FN={metrics['false_negative_coefficients']}",
        pad=4,
    )
    axis.axis("off")


def noise_plate(
    method_runs: list[SupplementaryRun],
    config: dict[str, Any],
) -> plt.Figure:
    """Create one complete diagnostic plate for a seed/noise pair."""
    time, clean, noisy, artifacts = _shared_test_data(method_runs)
    noise_level = method_runs[0].noise_level
    seed = method_runs[0].seed
    all_values = [clean, noisy]
    all_values.extend(artifact["test_prediction"] for artifact in artifacts)
    common_ylim = _padded_limits(all_values)

    figure = plt.figure(figsize=(15.5, 12.5))
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(
        1 + len(method_runs),
        4,
        height_ratios=[0.82] + [1.35] * len(method_runs),
        left=0.055,
        right=0.985,
        top=0.93,
        bottom=0.035,
        hspace=0.50,
        wspace=0.32,
    )

    noisy_axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    for trajectory_index, axis in enumerate(noisy_axes):
        _plot_noisy_test(axis, time, clean, noisy, trajectory_index)
        axis.set_ylim(*common_ylim)

    information_axis = figure.add_subplot(grid[0, 2:4])
    information_axis.axis("off")
    state_handles = [
        Line2D([], [], color=color, linewidth=1.6, label=rf"State ${state}$")
        for state, color in zip(STATE_NAMES, STATE_COLORS)
    ]
    style_handles = [
        Line2D([], [], color="#333333", linewidth=1.2, label="Clean reference"),
        Line2D(
            [],
            [],
            color="#333333",
            marker="o",
            markersize=3,
            linewidth=0,
            alpha=0.45,
            label="Noisy observation",
        ),
        Line2D(
            [],
            [],
            color="#333333",
            linewidth=1.2,
            linestyle=(0, (3.2, 2.0)),
            label="Selected-model rollout",
        ),
    ]
    information_axis.legend(
        handles=state_handles + style_handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        columnspacing=1.5,
    )
    information_axis.text(
        0.0,
        0.50,
        "Coefficient cells: green = true positive; orange = false positive; red = false negative.\n"
        "A dash denotes a coefficient below the configured support tolerance.\n"
        "Test data are shared across methods and used for reporting, not selection.\n"
        "Pareto color: DW deviation (lower is better); star: validation-selected model.\n"
        "Fit/DW references: noisy observations for Strong + SG/raw Weak; smoothed for Weak + SG.",
        ha="left",
        va="top",
        fontsize=7.6,
        color="#444444",
        linespacing=1.45,
    )

    for row, (run, artifact) in enumerate(zip(method_runs, artifacts), start=1):
        reconstruction_axes = [
            figure.add_subplot(grid[row, trajectory]) for trajectory in range(2)
        ]
        for trajectory_index, axis in enumerate(reconstruction_axes):
            _plot_reconstruction(
                axis,
                time,
                clean,
                artifact["test_prediction"],
                trajectory_index,
            )
            axis.set_ylim(*common_ylim)
        method_label = METHOD_LABELS[run.method]
        reconstruction_axes[0].text(
            -0.20,
            0.5,
            method_label,
            transform=reconstruction_axes[0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=METHOD_COLORS[run.method],
        )
        test_nrmse = run.payload["test"].get("rollout_nrmse")
        reconstruction_axes[1].text(
            0.98,
            0.96,
            "Test NRMSE=" + ("failed" if test_nrmse is None else f"{test_nrmse:.4g}"),
            transform=reconstruction_axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        _plot_pareto(figure.add_subplot(grid[row, 2]), run)
        _plot_coefficient_table(figure.add_subplot(grid[row, 3]), run, config)

    figure.suptitle(
        f"Noise {100 * noise_level:.0f}%  |  Seed {seed}",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    return figure



def joint_plate(run, config):
    """One page per repetition, with all five trajectories and the selected model."""
    arrays = run.arrays
    time = arrays["time"]
    fig = plt.figure(figsize=(12, 14))
    grid = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1.8],
                           left=.09, right=.98, bottom=.075, top=.94,
                           hspace=.75, wspace=.38)
    trajectories = [("train", 0), ("train", 1), ("validation", 0), ("test", 0), ("test", 1)]
    for row, (split, trajectory) in enumerate(trajectories):
        for state, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
            axis = fig.add_subplot(grid[row, state])
            clean = arrays[f"{split}_clean"][trajectory, :, state]
            noisy = arrays[f"{split}_noisy"][trajectory, :, state]
            prediction = arrays[f"{split}_prediction"][trajectory, :, state]
            axis.plot(time, clean, color="#222222", linewidth=1, label="Clean reference")
            axis.scatter(time[::10], noisy[::10], color=color, s=5, alpha=.35,
                         edgecolors="none", rasterized=True, label="Noisy observations")
            axis.plot(time, prediction, color=color, linestyle="--", linewidth=1,
                      label="Identified rollout")
            if not np.any(np.isfinite(prediction)):
                axis.text(.5, .5, "Rollout failed", transform=axis.transAxes, ha="center")
            axis.set_title(f"{split.capitalize()} {trajectory+1} - {name}", fontsize=9)
            _style_time_axis(axis, time, "State value" if state == 0 else None)
    _plot_pareto(fig.add_subplot(grid[5, :2]), run)
    _plot_coefficient_table(fig.add_subplot(grid[5, 2]), run, config)
    score = run.payload["test"].get("rollout_nrmse")
    score_text = "failed" if score is None else f"{score:.4g}"
    fig.text(.09, .985, f"Repetition {run.payload['job']['ic_block']} | Noise 10% | Test NRMSE {score_text}",
             ha="left", va="top", fontsize=12, fontweight="bold")
    handles = [Line2D([], [], color="#222222", label="Clean reference"),
               Line2D([], [], color="#555555", linestyle="--", label="Identified rollout"),
               Line2D([], [], color="#777777", marker="o", linestyle="none", label="Noisy observations")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.57,.98), fontsize=9)
    variables = run.payload["selected"]["variables"]
    fig.text(.09, .025,
             f"Selected library: Hill exponent {variables['gamma_1']:.4g}; exponential rate {variables['gamma_2']:.4g}. "
             "Fit/DW objectives use smoothed training observations.\n"
             "Coefficient cells: green = true positive; orange = false positive; red = false negative; dash = below support tolerance.",
             fontsize=7, ha="left", va="bottom", linespacing=1.5)
    return fig


def library_summary(payloads, config):
    """Use the archived validation-median rule and only active-family parameters."""
    ordered = sorted(payloads, key=lambda p: p["job"]["ic_block"])
    validation = [p["selected"]["validation"]["rollout_nrmse"] for p in ordered]
    median = float(np.median(validation))
    representative = min(ordered, key=lambda p: abs(p["selected"]["validation"]["rollout_nrmse"] - median))
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    parameters = {}
    for family, variable in (("Hill(", "gamma_1"), ("exp(", "gamma_2")):
        active = [p for p in ordered if any(
            _canonical_feature(name).startswith(family) and np.any(np.abs(row) > tolerance)
            for name, row in zip(p["selected"]["feature_names"], np.asarray(p["selected"]["coefficients"]))
        )]
        parameters[variable] = np.asarray([p["selected"]["variables"][variable] for p in active])
    return representative, parameters


def render_library_overview(raw, config, output):
    """Restore the compact trajectory/Hill/exponential figure using bundled data."""
    from benchmarks.common.data import trajectory_data

    payloads = [json.loads(path.read_text()) for path in sorted(raw.glob("*.json"))]
    representative, parameters = library_summary(payloads, config)
    with trajectory_data("joint") as trajectories, publication_style():
        arrays = trajectories[representative["job"]["job_id"]]
        time = arrays["time"]
        fig = plt.figure(figsize=(7.2, 5.2))
        grid = fig.add_gridspec(3, 2, width_ratios=[2.15, 1.55], hspace=.28, wspace=.34)
        for state, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
            axis = fig.add_subplot(grid[state, 0])
            axis.plot(time, arrays["test_clean"][0, :, state], color="#222222", label="Clean reference")
            axis.scatter(time[::10], arrays["test_noisy"][0, ::10, state], color=color,
                         s=8, alpha=.35, label="Noisy observations")
            axis.plot(time, arrays["test_prediction"][0, :, state], color=color,
                      linestyle="--", label="Identified rollout")
            axis.set_ylabel(name)
            axis.grid(axis="y", color="#D9D9D9", linewidth=.6)
            if state < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Time")
            if state == 0:
                handles, labels = axis.get_legend_handles_labels()

        axis = fig.add_subplot(grid[:2, 1])
        inputs = np.linspace(0, 2, 500)
        truth = float(config["truth"]["hill_exponent"])
        axis.plot(inputs, 1 / (1 + inputs**truth), color="#222222", linewidth=1.6,
                  label=rf"Ground truth ($\gamma={truth:.3g}$)")
        details = []
        for variable, family, color, linestyle, function in (
            ("gamma_1", "Hill", STATE_COLORS[2], "--", lambda g: 1 / (1 + inputs**g)),
            ("gamma_2", "Exponential", STATE_COLORS[0], ":", lambda g: np.exp(-g * inputs)),
        ):
            values = parameters[variable]
            details.append(f"{family} active: {len(values)}/{len(payloads)} runs")
            if not len(values):
                continue
            for value in values:
                axis.plot(inputs, function(value), color=color, linewidth=.7, alpha=.22)
            median = float(np.median(values))
            symbol = r"\gamma_1" if variable == "gamma_1" else r"\gamma_2"
            axis.plot(inputs, function(median), color=color, linewidth=1.6, linestyle=linestyle,
                      label=rf"{family} median (${symbol}={median:.3g}$)")
            details.append(rf"${symbol}$: median {median:.4g}; range [{min(values):.3g}, {max(values):.3g}]")
        axis.axvline(1, color="#BDBDBD", linewidth=.7, zorder=0)
        axis.set(xlabel="Library input", ylabel="Candidate value", xlim=(0, 2), ylim=(-.03, 1.03))
        axis.set_title("Optimized nonlinear candidates", loc="left")
        axis.grid(axis="y", color="#D9D9D9", linewidth=.6)
        axis.legend(loc="upper right", fontsize=7)
        detail_axis = fig.add_subplot(grid[2, 1])
        detail_axis.axis("off")
        details.extend([
            f"Example: repetition {representative['job']['ic_block']}, test trajectory 1",
            f"Test NRMSE {representative['test']['rollout_nrmse']:.3g}; "
            f"support F1 {representative['selected']['coefficient_metrics']['support_f1']:.3g}",
        ])
        detail_axis.text(0, .78, "\n".join(details), ha="left", va="top", fontsize=7.2, linespacing=1.45)
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.06, .955), ncol=3)
        output.mkdir(parents=True, exist_ok=True)
        stem = output / "figure_s4_joint_library_shapes"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')}")


def render(experiment, raw, config, output):
    from benchmarks.common.data import trajectory_data
    payloads = [json.loads(path.read_text()) for path in sorted(raw.glob("*.json"))]
    stem = "figure_s2_noise_diagnostics" if experiment == "noise" else "figure_s3_joint_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    with trajectory_data(experiment) as data, publication_style(), PdfPages(output / (stem + ".pdf")) as pdf:
        runs = [SupplementaryRun(p, data[p["job"]["job_id"]]) for p in payloads]
        if experiment == "noise":
            lookup = {(r.method, r.noise_level, r.seed): r for r in runs}
            groups = [[lookup[(method, float(noise), int(seed))] for method in DISPLAY_METHODS]
                      for noise in config["experiment"]["noise_levels"] for seed in config["experiment"]["seeds"]]
        else:
            groups = [[run] for run in sorted(runs, key=lambda r: r.payload["job"]["ic_block"])]
        for page, group in enumerate(groups, start=1):
            fig = noise_plate(group, config) if experiment == "noise" else joint_plate(group[0], config)
            pdf.savefig(fig, bbox_inches="tight")
            # Keep only a first-page preview with the selected PDF; all pages are in the PDF.
            if page == 1:
                fig.savefig(output / (stem + ".png"), dpi=160, bbox_inches="tight")
            plt.close(fig)
            pages.append({"page": page, "job_ids": [r.payload["job"]["job_id"] for r in group]})
    (output / (stem + "_pages.json")).write_text(json.dumps(pages, indent=2) + "\n")
    print(f"Wrote {len(pages)} diagnostic pages: {output / (stem + '.pdf')}")
    if experiment == "joint":
        render_library_overview(raw, config, output)
        from benchmarks.joint.feedback_comparison import (
            render as render_feedback_comparison,
        )
        render_feedback_comparison(raw, config, output)
