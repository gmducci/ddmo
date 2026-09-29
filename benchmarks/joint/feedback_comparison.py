"""Compare selected exponential and true Goodwin feedback along test trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from benchmarks.common.data import trajectory_data
from benchmarks.common.style import publication_style
from benchmarks.joint.postprocess import _canonical_feature

plt.switch_backend("Agg")

FIT_COLOR = "#3569A8"


def selected_exponentials(raw: Path, config: dict) -> list[dict]:
    """Read the fitted contribution in the x equation for each selected model."""
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    selected = []
    for path in sorted(raw.glob("*.json")):
        payload = json.loads(path.read_text())
        features = [_canonical_feature(name) for name in payload["selected"]["feature_names"]]
        index = features.index("exp(z)")
        coefficient = float(payload["selected"]["coefficients"][index][0])
        if abs(coefficient) > tolerance:
            selected.append({
                "job_id": payload["job"]["job_id"],
                "repetition": int(payload["job"]["ic_block"]),
                "coefficient": coefficient,
                "rate": float(payload["selected"]["variables"]["gamma_2"]),
            })
    return sorted(selected, key=lambda item: item["repetition"])


def normalize_cycles(
    reference: np.ndarray, fitted: np.ndarray, time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale both signals by one common maximum within each reference cycle."""
    step = float(time[1] - time[0])
    peaks, _ = find_peaks(
        reference,
        distance=max(1, int(round(15.0 / step))),
        prominence=0.005 * float(np.max(reference)),
    )
    if reference[0] >= reference[1]:
        peaks = np.r_[0, peaks]
    if not len(peaks):
        peaks = np.asarray([int(np.argmax(reference))])
    boundaries = [0]
    boundaries.extend(int((left + right) // 2) for left, right in zip(peaks[:-1], peaks[1:]))
    boundaries.append(len(reference))

    scaled_reference = np.empty_like(reference)
    scaled_fitted = np.empty_like(fitted)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        peak = max(float(np.max(reference[left:right])), float(np.max(fitted[left:right])))
        if peak <= 0:
            raise ValueError("Feedback pulse maximum must be positive")
        scaled_reference[left:right] = reference[left:right] / peak
        scaled_fitted[left:right] = fitted[left:right] / peak
    return scaled_reference, scaled_fitted


def render(raw: Path, config: dict, output: Path) -> None:
    """Render Figure S5 using the true and identified held-out rollouts."""
    models = selected_exponentials(raw, config)
    if not models:
        raise ValueError("No selected exponential feedback terms were found")

    truth = config["truth"]
    exponent = float(truth["hill_exponent"])
    half_saturation = float(truth["K1"])
    scale = float(truth["k1"])

    def true_feedback(z: np.ndarray) -> np.ndarray:
        return scale / (1.0 + (z / half_saturation) ** exponent)

    with trajectory_data("joint") as trajectories:
        for model in models:
            arrays = trajectories[model["job_id"]]
            model["time"] = arrays["time"]
            model["true_z"] = arrays["test_clean"][:, :, 2]
            model["identified_z"] = arrays["test_prediction"][:, :, 2]
            if not np.all(np.isfinite(model["identified_z"])):
                raise ValueError(f"Non-finite identified test rollout: {model['job_id']}")

    observed_truth = np.concatenate([
        true_feedback(model["true_z"]).ravel() for model in models
    ])
    observed_fit = np.concatenate([
        (model["coefficient"] * np.exp(-model["rate"] * model["identified_z"])).ravel()
        for model in models
    ])
    absolute_error = np.abs(observed_fit - observed_truth)
    print(
        f"Selected exponentials: {len(models)}; held-out time samples: {len(observed_truth)}; "
        f"median |feedback error|: {np.median(absolute_error):.4g}; "
        f"90th percentile: {np.quantile(absolute_error, 0.9):.4g}; "
        f"maximum: {np.max(absolute_error):.4g}"
    )

    with publication_style():
        fig, axes = plt.subplots(
            len(models), 2, figsize=(7.2, 7.8), sharex=True, sharey=True,
            squeeze=False,
        )
        for row, model in enumerate(models):
            true_values = true_feedback(model["true_z"])
            fitted_values = model["coefficient"] * np.exp(
                -model["rate"] * model["identified_z"]
            )
            time = model["time"]
            for trajectory in range(2):
                axis = axes[row, trajectory]
                scaled_true, scaled_fitted = normalize_cycles(
                    true_values[trajectory], fitted_values[trajectory], time,
                )
                axis.plot(
                    time, scaled_true,
                    color="#202020", linewidth=1.2,
                    label=r"True Hill on $z(t)$", zorder=3,
                )
                axis.plot(
                    time, scaled_fitted,
                    color=FIT_COLOR, linewidth=1.1,
                    label=r"Fitted exponential on $\hat z(t)$",
                    zorder=4,
                )
                axis.set(xlim=(time[0], time[-1]), ylim=(0, 1.04))
                axis.set_yticks([0, 0.5, 1.0])
                axis.set_xticks([0, 50, 100, 150, 200])
                axis.set_title(
                    f"Rep. {model['repetition']} · test trajectory {trajectory + 1}",
                    loc="left", pad=4,
                )
                axis.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.5)
        fig.supxlabel("Time", y=0.035)
        fig.supylabel("Cycle-normalized feedback", x=0.01)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   ncol=2, fontsize=8, columnspacing=1.6)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.99,
                            hspace=0.35, wspace=0.22)
        output.mkdir(parents=True, exist_ok=True)
        stem = output / "figure_s5_exponential_feedback"
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')}")
