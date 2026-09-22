"""Figure 2: selection frequency, conditional median coefficients, and ground truth."""
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from benchmarks.common.data import result_data
from benchmarks.common.goodwin import true_coefficient_matrix
from benchmarks.common.style import publication_style

plt.switch_backend("Agg")
STATE_NAMES = ["x", "y", "z"]


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


def render(raw, config, output):
    import json
    payloads = [(json.loads(path.read_text()), path.with_suffix(".npz")) for path in sorted(raw.glob("*.json"))]
    output.mkdir(parents=True, exist_ok=True)
    plot_selection_frequency(output / "figure_2_library_recovery", payloads, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path(".local/benchmark_plots/joint"))
    args = parser.parse_args()
    with result_data("joint", args.raw_directory) as (config, raw):
        render(raw, config, args.output_directory)
    print(f"Wrote Figure 2 to {args.output_directory}")


if __name__ == "__main__":
    main()
