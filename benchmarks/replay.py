"""Replay one frozen Goodwin trajectory from its selected coefficients.

For example, from the repository root::

    python -m benchmarks.replay noise --method weak_optimized --noise 0.10 --seed 11
    python -m benchmarks.replay joint --noise 0.10 --seed 104 --split test --index 1

This reads the frozen result and time grid. It does not regenerate data or fit a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pysindy as ps
import tomllib
from scipy.integrate import solve_ivp

from benchmarks.common.goodwin import STATE_NAMES, build_library

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "benchmarks/data/current"


def replay(
    experiment: str,
    *,
    noise: float,
    seed: int,
    method: str | None = None,
    split: str = "test",
    index: int = 0,
    data_root: Path = CURRENT,
) -> tuple[str, np.ndarray, np.ndarray, float]:
    """Return job ID, time, integrated states, and maximum saved-point error."""
    if experiment not in {"noise", "joint"}:
        raise ValueError(f"Unknown experiment: {experiment}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown trajectory split: {split}")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("Noise must be finite and non-negative")
    if index < 0:
        raise ValueError("Trajectory index must be non-negative")

    folder = data_root / experiment
    matches = []
    for path in sorted((folder / "raw").glob("*.json")):
        payload = json.loads(path.read_text())
        job = payload["job"]
        if (
            job["data_seed"] == seed
            and math.isclose(job["noise_level"], noise, rel_tol=0, abs_tol=1e-12)
            and (method is None or job["method"] == method)
        ):
            matches.append((path, payload))
    if not matches:
        raise ValueError(
            f"No frozen {experiment} run for method={method}, noise={noise}, seed={seed}"
        )
    if len(matches) != 1:
        methods = sorted({payload["job"]["method"] for _, payload in matches})
        raise ValueError(f"Selection matches {len(matches)} runs; specify --method from {methods}")

    path, payload = matches[0]
    if payload["status"] != "complete" or payload["mode"] != "production":
        raise ValueError(f"Run is not a completed production result: {path}")
    job_id = payload["job"]["job_id"]
    if path.stem != job_id:
        raise ValueError(f"Run filename does not match job ID: {path}")
    selected = payload["selected"]
    coefficients = np.asarray(selected["coefficients"], dtype=float)
    feature_names = selected["feature_names"]
    if coefficients.shape != (len(feature_names), len(STATE_NAMES)) or not np.all(
        np.isfinite(coefficients)
    ):
        raise ValueError(f"Invalid selected coefficient matrix: {job_id}")

    config = tomllib.loads((folder / "config.toml").read_text())
    library, _ = build_library(config, adaptive_exponent=experiment == "joint")
    named_functions = library.build(selected["variables"])
    feature_library = ps.CustomLibrary(
        library_functions=[function for function, _ in named_functions],
        function_names=[name for _, name in named_functions],
    )

    counts = payload["data"]["split_counts"]
    if index >= counts[split]:
        raise ValueError(f"{split} index must be between 0 and {counts[split] - 1}")
    splits = ("train", "validation", "test")
    offset = sum(counts[part] for part in splits[:splits.index(split)])
    initial_condition = np.asarray(payload["data"]["initial_conditions"][offset + index], dtype=float)
    if initial_condition.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(initial_condition)):
        raise ValueError(f"Invalid initial condition: {job_id}")
    feature_library.fit(np.atleast_2d(initial_condition))
    actual_names = list(feature_library.get_feature_names(STATE_NAMES))
    if actual_names != feature_names:
        raise ValueError(f"Feature order differs from saved coefficient matrix: {job_id}")

    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        time = archive["time"].copy()
        saved = archive[f"{split}_prediction"][index].copy()
        if "initial_conditions" in archive:
            np.testing.assert_array_equal(archive["initial_conditions"][offset + index], initial_condition)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError(f"Invalid saved time grid: {job_id}")
    if saved.shape != (len(time), len(STATE_NAMES)):
        raise ValueError(f"Invalid saved prediction shape: {job_id}")

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(feature_library.transform(np.atleast_2d(state)) @ coefficients).ravel()

    def divergence_event(_time: float, state: np.ndarray) -> float:
        return 1e10 - np.max(np.abs(state))

    divergence_event.terminal = True
    divergence_event.direction = -1
    solution = solve_ivp(
        rhs,
        (time[0], time[-1]),
        initial_condition,
        t_eval=time,
        method="BDF",
        rtol=1e-7,
        atol=1e-9,
        events=divergence_event,
    )
    if not solution.success or len(solution.t) != len(time):
        raise RuntimeError(f"Integration failed or diverged for {job_id}: {solution.message}")
    trajectory = solution.y.T
    if not np.all(np.isfinite(saved)):
        raise ValueError(f"Saved {split} prediction is non-finite: {job_id}")
    max_error = float(np.max(np.abs(trajectory - saved)))
    np.testing.assert_allclose(trajectory, saved, rtol=1e-7, atol=1e-8,
                               err_msg=f"Replayed trajectory differs from saved prediction: {job_id}")
    return job_id, time, trajectory, max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("noise", "joint"))
    parser.add_argument("--method", help="Required when seed and noise match multiple methods")
    parser.add_argument("--noise", type=float, required=True, help="Fractional noise level, e.g. 0.10")
    parser.add_argument("--seed", type=int, required=True, help="Data seed identifying the frozen run")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--index", type=int, default=0, help="Zero-based trajectory index within the split")
    parser.add_argument("--output", type=Path, help="CSV output path; defaults under .local/benchmark_replay")
    args = parser.parse_args()

    try:
        job_id, time, trajectory, max_error = replay(
            args.experiment, noise=args.noise, seed=args.seed, method=args.method,
            split=args.split, index=args.index,
        )
    except (ValueError, RuntimeError, AssertionError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")
    output = args.output or ROOT / ".local/benchmark_replay" / args.experiment / (
        f"{job_id}-{args.split}{args.index}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", *STATE_NAMES))
        writer.writerows(np.column_stack((time, trajectory)))
    print(f"Replayed {job_id} {args.split}[{args.index}] ({len(time)} points)")
    print(f"Maximum difference from saved prediction: {max_error:.3g}")
    print(f"Trajectory: {output}")


if __name__ == "__main__":
    main()
