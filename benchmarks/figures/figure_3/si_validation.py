"""Plot every selected Figure 3 validation rollout for supplementary material."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tomllib
from matplotlib.lines import Line2D

from benchmarks.figures.figure_3.postprocess import collect_payloads

plt.switch_backend("Agg")

STATE_NAMES = [r"$x$", r"$y$", r"$z$"]
STATE_COLORS = ["#3B4CC0", "#00876C", "#B40426"]
STATE_OFFSETS = np.asarray([2.4, 1.2, 0.0])


def _normalized_validation(
    payload: dict[str, Any], artifact_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.load(artifact_path)
    time = np.asarray(arrays["time"], dtype=float)
    clean = np.asarray(arrays["validation_clean"][0], dtype=float)
    prediction = np.asarray(arrays["validation_prediction"][0], dtype=float)
    lower = np.min(clean, axis=0)
    scale = np.ptp(clean, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    clean_normalized = (clean - lower) / scale
    prediction_normalized = (prediction - lower) / scale
    return time, clean_normalized, prediction_normalized


def plot_all_validation_rollouts(
    output_stem: Path,
    payloads: list[tuple[dict[str, Any], Path]],
) -> None:
    """Create the 10-block by 3-training-size supplementary plate."""
    lookup = {
        (
            int(payload["job"]["ic_block"]),
            int(payload["job"]["training_trajectories"]),
        ): (payload, artifact)
        for payload, artifact in payloads
    }
    blocks = sorted({block for block, _ in lookup})
    training_sizes = sorted({size for _, size in lookup})
    expected = {(block, size) for block in blocks for size in training_sizes}
    missing = sorted(expected - set(lookup))
    if missing:
        raise RuntimeError(f"Missing validation results for: {missing}")

    figure, axes = plt.subplots(
        len(blocks),
        len(training_sizes),
        figsize=(9.0, 14.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    figure.patch.set_facecolor("white")

    for row, block in enumerate(blocks):
        for column, training_size in enumerate(training_sizes):
            axis = axes[row, column]
            payload, artifact = lookup[(block, training_size)]
            time, clean, prediction = _normalized_validation(payload, artifact)
            step = max(1, len(time) // 800)
            for state_index, (color, offset) in enumerate(
                zip(STATE_COLORS, STATE_OFFSETS)
            ):
                axis.plot(
                    time[::step],
                    clean[::step, state_index] + offset,
                    color=color,
                    linewidth=0.70,
                    solid_capstyle="round",
                    zorder=2,
                )
                axis.plot(
                    time[::step],
                    prediction[::step, state_index] + offset,
                    color=color,
                    linewidth=0.82,
                    linestyle=(0, (3.0, 2.0)),
                    solid_capstyle="round",
                    zorder=3,
                )

            nrmse = float(payload["selected"]["validation"]["rollout_nrmse"])
            axis.text(
                0.985,
                0.94,
                f"NRMSE={nrmse:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.4,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 0.8,
                },
                zorder=5,
            )
            axis.set_xlim(float(time[0]), float(time[-1]))
            axis.set_ylim(-0.08, 3.58)
            axis.set_yticks(STATE_OFFSETS + 0.5)
            if column == 0:
                axis.set_yticklabels(STATE_NAMES, fontsize=7)
                for label, color in zip(axis.get_yticklabels(), STATE_COLORS):
                    label.set_color(color)
                axis.text(
                    -0.24,
                    0.5,
                    f"Block {block:02d}",
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#333333",
                )
            else:
                axis.tick_params(labelleft=False)
            if row == 0:
                axis.set_title(
                    rf"$N_{{\mathrm{{train}}}}={training_size}$",
                    fontsize=9,
                    fontweight="bold",
                    pad=5,
                )
            if row == len(blocks) - 1:
                axis.set_xticks([0, 100, 200])
                axis.set_xlabel("Time", fontsize=7.5)
            else:
                axis.tick_params(labelbottom=False)
            axis.tick_params(axis="x", labelsize=6.5, length=2.5, width=0.6)
            axis.tick_params(axis="y", length=0)
            axis.grid(axis="x", color="#E6E6E6", linewidth=0.45)
            for spine in axis.spines.values():
                spine.set_color("#B8B8B8")
                spine.set_linewidth(0.55)

    legend = [
        Line2D([0], [0], color="#333333", linewidth=1.1, label="Clean reference"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=1.1,
            linestyle=(0, (3.0, 2.0)),
            label="Selected-model validation rollout",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.951),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handlelength=2.8,
        columnspacing=1.6,
    )
    figure.suptitle(
        "Supplementary Figure S1 | All Figure 3 validation reconstructions",
        x=0.075,
        y=0.989,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.967,
        "Rows are paired Latin-hypercube blocks; columns are nested training-set sizes. "
        "Each state is independently range-normalized using its clean validation trajectory and vertically offset.",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.905,
        bottom=0.045,
        hspace=0.26,
        wspace=0.08,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--raw-directory", type=Path)
    parser.add_argument("--output", type=Path, help="Output stem without extension")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    config_path = (
        args.config.resolve() if args.config else Path(__file__).with_name("config.toml")
    )
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    experiment = str(config["experiment"]["name"])
    raw_directory = args.raw_directory or (
        repo_root / "benchmarks/results/raw" / experiment
    )
    output_stem = args.output or (
        repo_root
        / "benchmarks/results/figures/figure_si_validation_reconstructions"
    )
    payloads = collect_payloads(raw_directory.resolve(), experiment)
    plot_all_validation_rollouts(output_stem.resolve(), payloads)
    print(f"processed {len(payloads)} validation rollouts")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {output_stem.with_suffix('.svg')}")
    print(f"wrote {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
