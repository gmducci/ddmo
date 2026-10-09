"""Render all selected corrected campaign models on A4 supplementary pages.

Figure 1 has one page per noise/seed pair; Figure 2 has one page per block.
The pages use saved campaign results only. They do not rerun optimization.
Run ``python -m benchmarks.si_paper`` from the repository root.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tomllib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from benchmarks.common.goodwin import true_coefficient_matrix
from benchmarks.common.style import publication_style
from benchmarks.noise.postprocess import feature_label

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
CURRENT = ROOT / "benchmarks/data/current"
OUTPUT = LOCAL / "si_paper_render"
A4_INCHES = (8.2677165, 11.6929134)
PDF_METADATA = {
    "CreationDate": datetime(2026, 10, 9, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 10, 9, tzinfo=timezone.utc),
}
STATE_NAMES = ("x", "y", "z")
SUPPORT_COLORS = ListedColormap(("#FFFFFF", "#CFE7DA", "#F3D5AF", "#F2C9CE"))


def _read_config(folder: Path):
    return tomllib.loads((folder / "config.toml").read_text())


def _verify_campaign(folder: Path):
    source = json.loads((folder / "source.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    if report["status"] != "complete_local_preview":
        raise ValueError(f"Campaign is incomplete: {folder}")
    for path, key in ((ROOT / "src/ddmo/formulation/weak.py", "weak_py_sha256"),
                      (folder / "config.toml", "config_sha256")):
        if _digest(path) != source[key] or source[key] != report[key]:
            raise ValueError(f"Campaign provenance differs from {path}")


def _page():
    fig = plt.figure(figsize=A4_INCHES, facecolor="white")
    return fig


def _support_axis(axis, names, values, truth, tolerance, title, *, row_labels=False):
    active = np.abs(values) > tolerance
    expected = np.abs(truth) > tolerance
    codes = np.zeros(values.shape, dtype=int)
    codes[active & expected] = 1
    codes[active & ~expected] = 2
    codes[~active & expected] = 3
    axis.imshow(codes, cmap=SUPPORT_COLORS, vmin=0, vmax=3,
                aspect="auto", interpolation="none")
    for row, column in np.ndindex(values.shape):
        if active[row, column]:
            axis.text(column, row, f"{values[row, column]:.2g}",
                      ha="center", va="center", fontsize=6.4, color="#252525")
        elif expected[row, column]:
            axis.text(column, row, "0", ha="center", va="center",
                      fontsize=6.4, color="#772F38")
    axis.set_xticks(range(3), [rf"$\dot{{{state}}}$" for state in STATE_NAMES])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=3, labelsize=9)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels([feature_label(name) for name in names] if row_labels else [])
    axis.tick_params(axis="y", length=0, pad=5, labelsize=8)
    axis.set_xticks(np.arange(-.5, 3, 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    axis.grid(which="minor", color="#D8D8D8", linewidth=.45)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_title(title, fontsize=9, fontweight="bold", pad=25)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_campaign_runs(folder: Path) -> list[dict]:
    runs = [json.loads(path.read_text()) for path in sorted((folder / "raw").glob("*.json"))]
    for run in runs:
        if run["status"] != "complete" or run["mode"] != "production":
            raise ValueError(f"Incomplete selected model: {run['job']['job_id']}")
    return runs


def _artifact(folder: Path, run: dict) -> dict[str, np.ndarray]:
    path = folder / "raw" / (run["job"]["job_id"] + ".npz")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _strong_artifacts(folder: Path, runs: list[dict]) -> None:
    """Require the unchanged strong baseline trajectories in the frozen campaign."""
    strong = [run for run in runs if run["job"]["method"] == "strong"]
    missing = [run for run in strong if not (
        folder / "raw" / (run["job"]["job_id"] + ".npz")).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} strong baseline trajectories")


def _trajectory(axis, time, clean, noisy, prediction, color, show_time):
    axis.plot(time, clean, color="#555555", linestyle="--", linewidth=.9, zorder=2)
    if np.any(np.isfinite(prediction)):
        axis.plot(time, prediction, color=color, linewidth=1.45, zorder=3)
    else:
        axis.text(.5, .5, "Rollout failed", transform=axis.transAxes,
                  ha="center", va="center", fontsize=8)
    step = max(1, len(time) // 55)
    axis.scatter(time[::step], noisy[::step], s=12, facecolor="#202020",
                 edgecolor="white", linewidth=.2, alpha=.9, zorder=4,
                 rasterized=True)
    axis.set_xlim(time[0], time[-1])
    axis.grid(axis="y", color="#E5E5E5", linewidth=.45)
    axis.tick_params(labelsize=7, pad=2)
    if show_time:
        axis.set_xlabel("Time", fontsize=8)
    else:
        axis.tick_params(labelbottom=False)


def _plot_legend(fig, *, selected=False, y=.477):
    handles = [
        Line2D([], [], color="#555555", linestyle="--", linewidth=1,
               label="Clean reference"),
        Line2D([], [], color="#202020", marker="o", linestyle="none",
               markersize=4, label="Noisy observations"),
    ]
    if selected:
        handles.append(Line2D([], [], color="#B40426", linewidth=1.5,
                              label="Selected model"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, y),
               ncol=len(handles), frameon=False, fontsize=8)


def _noise_page(config, group, folder, noise, seed):
    methods = ("strong", "weak_optimized", "weak_optimized_sg")
    labels = {"strong": "Strong + SG", "weak_optimized": "Weak",
              "weak_optimized_sg": "Weak + SG"}
    colors = {"strong": "#3B4CC0", "weak_optimized": "#B40426",
              "weak_optimized_sg": "#D8751A"}
    arrays = {method: _artifact(folder, group[method]) for method in methods}
    reference = arrays["weak_optimized"]
    for method in methods:
        for key in ("time", "test_clean", "test_noisy"):
            if not np.array_equal(reference[key], arrays[method][key]):
                raise ValueError(f"Figure 1 test data differ: {noise}, {seed}, {method}, {key}")
    names = group["strong"]["selected"]["feature_names"]
    if any(group[method]["selected"]["feature_names"] != names for method in methods):
        raise ValueError(f"Figure 1 feature order differs: {noise}, {seed}")
    truth = true_coefficient_matrix(names, config)
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    fig = _page()
    fig.text(.155, .958, f"Figure S1  |  {noise:.0%} noise, seed {seed}",
             fontsize=11, fontweight="bold", ha="left")
    fig.text(.155, .936, "Selected coefficient matrices and first held-out trajectory",
             fontsize=8, color="#444444", ha="left")
    outer = fig.add_gridspec(2, 1, left=.155, right=.965, top=.875, bottom=.135,
                             height_ratios=(1.14, .86), hspace=.20)
    tables = outer[0].subgridspec(1, 4, wspace=.14)
    panels = [("Truth", truth)] + [
        (labels[method], np.asarray(group[method]["selected"]["coefficients"], dtype=float))
        for method in methods
    ]
    for index, (title, matrix) in enumerate(panels):
        _support_axis(fig.add_subplot(tables[index]), names, matrix, truth,
                      tolerance, title, row_labels=index == 0)
    fig.text(.105, .875, "a", fontweight="bold", fontsize=10, va="top")
    plots = outer[1].subgridspec(3, 3, hspace=.17, wspace=.20)
    time = reference["time"]
    for col, method in enumerate(methods):
        for row, state in enumerate(("x", "y", "z")):
            axis = fig.add_subplot(plots[row, col])
            _trajectory(axis, time, reference["test_clean"][0, :, row],
                        reference["test_noisy"][0, :, row],
                        arrays[method]["test_prediction"][0, :, row],
                        colors[method], show_time=row == 2)
            if col == 0:
                axis.set_ylabel(rf"${state}$", rotation=0, labelpad=10)
            if row == 0:
                axis.set_title(labels[method], fontsize=8, pad=5, color=colors[method])
    fig.text(.105, .45, "b", fontweight="bold", fontsize=10, va="top")
    _plot_legend(fig)
    fig.text(.155, .075,
             "Green = true positive; orange = false positive; rose = missed term. "
             "Numbers show coefficients.", fontsize=7.3)
    fig.text(.155, .053,
             "Dots show noisy test observations; colored lines show selected-model rollouts.",
             fontsize=7.3)
    return fig


def render_noise():
    folder = CURRENT / "noise"
    _verify_campaign(folder)
    config = _read_config(folder)
    runs = _load_campaign_runs(folder)
    _strong_artifacts(folder, runs)
    lookup = {(run["job"]["method"], float(run["job"]["noise_level"]),
               int(run["job"]["data_seed"])): run for run in runs}
    methods = ("strong", "weak_optimized", "weak_optimized_sg")
    levels = [float(value) for value in config["experiment"]["noise_levels"]]
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    expected = {(method, noise, seed) for method in methods
                for noise in levels for seed in seeds}
    if len(runs) != len(expected) or set(lookup) != expected:
        raise ValueError(f"Figure 1 run matrix differs: {set(lookup) ^ expected}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "figure_s1_noise_all_models_a4"
    pages = []
    with publication_style():
        plt.rcParams["savefig.bbox"] = None
        with PdfPages(stem.with_suffix(".pdf"), metadata=PDF_METADATA) as pdf:
            for noise in levels:
                for seed in seeds:
                    group = {method: lookup[method, noise, seed] for method in methods}
                    fig = _noise_page(config, group, folder, noise, seed)
                    pdf.savefig(fig, bbox_inches=None)
                    if not pages:
                        fig.savefig(stem.with_suffix(".png"), dpi=170, bbox_inches=None)
                    plt.close(fig)
                    pages.append({"page": len(pages) + 1, "noise_level": noise,
                                  "seed": seed,
                                  "job_ids": [group[method]["job"]["job_id"]
                                              for method in methods]})
    stem.with_suffix(".json").write_text(json.dumps({
        "status": "current_paper_result", "page_size": "A4 portrait", "pages": pages,
        "plot_test_trajectory": 1, "config_sha256": _digest(folder / "config.toml"),
        "weak_source_sha256": _digest(ROOT / "src/ddmo/formulation/weak.py"),
    }, indent=2) + "\n")
    return stem.with_suffix(".pdf")


def _joint_page(config, payload, arrays):
    job = payload["job"]
    names = payload["selected"]["feature_names"]
    estimate = np.asarray(payload["selected"]["coefficients"], dtype=float)
    truth = true_coefficient_matrix(names, config)
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    fig = _page()
    fig.text(.15, .958,
             f"Figure S2  |  {job['noise_level']:.0%} noise, seed {job['data_seed']}",
             fontsize=11, fontweight="bold", ha="left")
    fig.text(.15, .936, f"Joint-library run, block {job['ic_block']}; two held-out trajectories",
             fontsize=8, color="#444444", ha="left")
    outer = fig.add_gridspec(2, 1, left=.15, right=.965, top=.875, bottom=.135,
                             height_ratios=(1.16, .84), hspace=.24)
    tables = outer[0].subgridspec(1, 2, wspace=.22)
    for index, (title, matrix) in enumerate((("Truth", truth), ("Selected model", estimate))):
        _support_axis(fig.add_subplot(tables[index]), names, matrix, truth,
                      tolerance, title, row_labels=index == 0)
    fig.text(.105, .875, "a", fontweight="bold", fontsize=10, va="top")
    plots = outer[1].subgridspec(3, 2, hspace=.17, wspace=.20)
    time = arrays["time"]
    for row, state in enumerate(("x", "y", "z")):
        for col in range(2):
            axis = fig.add_subplot(plots[row, col])
            _trajectory(axis, time, arrays["test_clean"][col, :, row],
                        arrays["test_noisy"][col, :, row],
                        arrays["test_prediction"][col, :, row], "#B40426",
                        show_time=row == 2)
            if col == 0:
                axis.set_ylabel(rf"${state}$", rotation=0, labelpad=10)
            if row == 0:
                axis.set_title(f"Held-out trajectory {col + 1}", fontsize=8, pad=5)
    fig.text(.105, .447, "b", fontweight="bold", fontsize=10, va="top")
    _plot_legend(fig, selected=True, y=.472)
    fig.text(.15, .075,
             "Green = true positive; orange = false positive; rose = missed term. "
             "Numbers show coefficients.", fontsize=7.3)
    fig.text(.15, .053,
             f"Test NRMSE {payload['test']['rollout_nrmse']:.3g}; "
             f"selected Hill exponent {payload['selected']['variables']['gamma_1']:.3g} "
             f"(truth {config['truth']['hill_exponent']:.3g}); "
             f"exponential rate {payload['selected']['variables']['gamma_2']:.3g}.",
             fontsize=7.3)
    return fig


def render_joint():
    folder = CURRENT / "joint"
    _verify_campaign(folder)
    config = _read_config(folder)
    runs = sorted(_load_campaign_runs(folder), key=lambda run: run["job"]["ic_block"])
    if [run["job"]["ic_block"] for run in runs] != list(range(1, 11)):
        raise ValueError("Figure 2 blocks differ from the ten-run campaign")
    stem = OUTPUT / "figure_s2_joint_all_models_a4"
    pages = []
    with publication_style():
        plt.rcParams["savefig.bbox"] = None
        with PdfPages(stem.with_suffix(".pdf"), metadata=PDF_METADATA) as pdf:
            for run in runs:
                fig = _joint_page(config, run, _artifact(folder, run))
                pdf.savefig(fig, bbox_inches=None)
                if not pages:
                    fig.savefig(stem.with_suffix(".png"), dpi=170, bbox_inches=None)
                plt.close(fig)
                pages.append({"page": len(pages) + 1,
                              "noise_level": run["job"]["noise_level"],
                              "seed": run["job"]["data_seed"],
                              "block": run["job"]["ic_block"],
                              "job_id": run["job"]["job_id"]})
    stem.with_suffix(".json").write_text(json.dumps({
        "status": "current_paper_result", "page_size": "A4 portrait", "pages": pages,
        "plot_test_trajectories": [1, 2],
        "config_sha256": _digest(folder / "config.toml"),
        "weak_source_sha256": _digest(ROOT / "src/ddmo/formulation/weak.py"),
    }, indent=2) + "\n")
    return stem.with_suffix(".pdf")


def main():
    print(render_noise())
    print(render_joint())


if __name__ == "__main__":
    main()
