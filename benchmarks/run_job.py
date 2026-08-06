"""Run one deterministic ddmo publication-benchmark job.

The first supported experiment is ``feature_recovery``.  Smoke mode follows
the same execution and reporting path with a deliberately tiny data and
optimizer budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ddmo import (
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.equations.brussellator import Brussellator
from ddmo.library_functions import LibraryBuilder
from ddmo.optimizer import make_active_coefficient_count, trajectory_mse
from ddmo.utils.ode_data_generator import ODEDataBatch, ODEDataGenerator

try:
    from benchmarks.plan_runs import build_jobs, load_config
except ModuleNotFoundError:  # Support ``python benchmarks/run_job.py``.
    from plan_runs import build_jobs, load_config


SMOKE_POPULATION = 12
SMOKE_GENERATIONS = 3
SMOKE_DURATION = 2.0
STATE_NAMES = ["x", "y"]
DEFAULT_INITIAL_CONDITIONS = [
    [1.0, 1.0],
    [1.2, 2.5],
    [2.0, 1.5],
    [0.8, 3.0],
    [1.5, 1.0],
    [2.5, 3.0],
]


@dataclass(frozen=True)
class RuntimePlan:
    """Resolved job settings after applying optional smoke overrides."""

    job: dict[str, Any]
    population: int
    generations: int
    t_end: float
    split_counts: tuple[int, int, int]
    mode: str

    @property
    def evaluation_budget(self) -> int:
        return self.population * self.generations


def make_runtime_plan(config: dict, job: dict, *, smoke: bool) -> RuntimePlan:
    """Resolve production or smoke settings without mutating the TOML data."""
    data = config["data"]
    if smoke:
        return RuntimePlan(
            job=job,
            population=SMOKE_POPULATION,
            generations=SMOKE_GENERATIONS,
            t_end=min(float(data["t_end"]), float(data["t_start"]) + SMOKE_DURATION),
            split_counts=(1, 1, 1),
            mode="smoke",
        )

    return RuntimePlan(
        job=job,
        population=int(config["optimizer"]["population"]),
        generations=int(config["optimizer"]["generations"]),
        t_end=float(data["t_end"]),
        split_counts=(
            int(data["train_trajectories"]),
            int(data["validation_trajectories"]),
            int(data["test_trajectories"]),
        ),
        mode="production",
    )


def _build_library(method: str, config: dict) -> tuple[LibraryBuilder, dict]:
    """Build the common four-term Brusselator candidate library."""
    if method == "adaptive_library":
        exponent_source = None
        library_bounds = {
            name: tuple(bounds) for name, bounds in config["bounds"]["library"].items()
        }
    elif method == "fixed_library":
        exponent_source = {
            "g1": float(config["fixed_library"]["g1"]),
            "g2": float(config["fixed_library"]["g2"]),
        }
        library_bounds = {}
    else:
        raise ValueError(f"Unsupported feature-recovery method: {method}")

    def constant(x, y, params):
        del y, params
        return np.ones_like(x)

    def interaction(x, y, params):
        exponents = exponent_source if exponent_source is not None else params
        return (x ** exponents["g1"]) * (y ** exponents["g2"])

    def interaction_name(x, y, params):
        exponents = exponent_source if exponent_source is not None else params
        return f"{x}^{exponents['g1']:.6g}*{y}^{exponents['g2']:.6g}"

    library = (
        LibraryBuilder()
        .add_identity()
        .add_custom(constant, "1")
        .add_custom(interaction, interaction_name)
    )
    return library, library_bounds


def _initial_conditions(config: dict, required: int) -> list[np.ndarray]:
    configured = config["data"].get("initial_conditions", DEFAULT_INITIAL_CONDITIONS)
    if len(configured) < required:
        raise ValueError(
            f"data.initial_conditions needs at least {required} entries; "
            f"found {len(configured)}"
        )
    return [np.asarray(values, dtype=float) for values in configured[:required]]


def _generate_data(config: dict, plan: RuntimePlan) -> ODEDataBatch:
    data = config["data"]
    truth = config["truth"]
    n_train, n_validation, n_test = plan.split_counts
    initial_conditions = _initial_conditions(config, n_train + n_validation + n_test)
    system = Brussellator(
        a=float(truth.get("a", 1.0)),
        b=float(truth.get("b", 3.0)),
        alpha=float(truth["g1"]),
        beta=float(truth["g2"]),
    )
    dt = float(data["dt"])
    time_array = np.arange(float(data["t_start"]), plan.t_end + 0.5 * dt, dt)
    return ODEDataGenerator(
        system.dynamics,
        initial_conditions=initial_conditions,
        time_array=time_array,
        noise_level=float(plan.job["noise_level"]),
        seed=int(plan.job["data_seed"]),
        clip_min=1e-12,
    ).generate()


def _split_batch(batch: ODEDataBatch, counts: tuple[int, int, int]):
    n_train, n_validation, n_test = counts
    validation_end = n_train + n_validation
    return (
        slice(0, n_train),
        slice(n_train, validation_end),
        slice(validation_end, validation_end + n_test),
    )


def _rollout_metrics(model, batch: ODEDataBatch, split: slice) -> dict[str, Any]:
    references = batch.x_clean[split]
    initial_conditions = batch.initial_conditions[split]
    predictions = []
    try:
        for initial_condition in initial_conditions:
            predictions.append(model.simulate(initial_condition, batch.time))
    except Exception as exc:
        return {
            "rollout_nrmse": float("inf"),
            "per_state_nrmse": [float("inf")] * references.shape[-1],
            "simulation_error": f"{type(exc).__name__}: {exc}",
        }

    prediction_array = np.asarray(predictions)
    axes = (0, 1)
    rmse = np.sqrt(np.mean((prediction_array - references) ** 2, axis=axes))
    scale = np.std(references, axis=axes)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    per_state = rmse / scale
    return {
        "rollout_nrmse": float(np.mean(per_state)),
        "per_state_nrmse": per_state.tolist(),
        "simulation_error": None,
    }


def _truth_coefficients(config: dict) -> np.ndarray:
    """Return coefficients for [x, y, 1, x^g1*y^g2] by [dx, dy]."""
    a = float(config["truth"].get("a", 1.0))
    b = float(config["truth"].get("b", 3.0))
    return np.asarray(
        [
            [-(b + 1.0), b],
            [0.0, 0.0],
            [1.0, 0.0],
            [a, -a],
        ]
    )


def _coefficient_metrics(
    coefficients: np.ndarray, config: dict, tolerance: float
) -> dict[str, float]:
    estimated = np.asarray(coefficients, dtype=float)
    truth = _truth_coefficients(config)
    if estimated.shape != truth.shape:
        return {
            "relative_l2_error": float("inf"),
            "support_accuracy": 0.0,
            "active_coefficients": float(
                np.count_nonzero(np.abs(estimated) > tolerance)
            ),
        }
    truth_support = np.abs(truth) > tolerance
    estimated_support = np.abs(estimated) > tolerance
    return {
        "relative_l2_error": float(
            np.linalg.norm(estimated - truth) / np.linalg.norm(truth)
        ),
        "support_accuracy": float(np.mean(estimated_support == truth_support)),
        "active_coefficients": float(np.count_nonzero(estimated_support)),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"git_commit": commit, "git_dirty": dirty}


def _package_versions() -> dict[str, str]:
    packages = ["ddmo", "numpy", "scipy", "scikit-learn", "pysindy", "pymoo"]
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run_feature_recovery(
    config: dict,
    plan: RuntimePlan,
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Execute and summarize one feature-recovery job."""
    started = time.perf_counter()
    batch = _generate_data(config, plan)
    train_slice, validation_slice, test_slice = _split_batch(batch, plan.split_counts)
    training = ODEDataBatch(
        time=batch.time,
        x_clean=batch.x_clean[train_slice],
        x_noisy=batch.x_noisy[train_slice],
        x_dot_clean=batch.x_dot_clean[train_slice],
        initial_conditions=batch.initial_conditions[train_slice],
    )
    data_instance = training.to_data_constructor(include_derivatives=True)

    library, library_bounds = _build_library(plan.job["method"], config)
    solver_bounds = {
        name: tuple(bounds) for name, bounds in config["bounds"]["solver"].items()
    }
    optimizer_settings = OptimizerSettings(
        library_bounds=library_bounds,
        formulation_bounds={},
        solver_bounds=solver_bounds,
        n_gen=plan.generations,
        pop_size=plan.population,
        algorithm=config["optimizer"]["algorithm"],
        seed=int(plan.job["optimizer_seed"]),
        verbose=False,
    )
    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "fit_intercept": False,
            "normalize_columns": False,
        },
        scale_features=True,
    )
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    objectives = {}
    for name in config["optimizer"]["objectives"]:
        if name == "trajectory_mse":
            objectives[name] = trajectory_mse
        elif name in {"active_coefficients", "active_coefficient_count"}:
            objectives[name] = make_active_coefficient_count(tolerance=tolerance)
        else:
            raise ValueError(f"Unsupported objective in feature recovery: {name}")

    result = identify(
        data_instance=data_instance,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=FormulationSettings(formulation="strong"),
        library_object=library.build,
        objective_functions=objectives,
    )
    if len(result.models) == 0:
        raise RuntimeError("Optimization returned no Pareto solutions.")

    objective_names = list(objectives)
    variable_names = list(optimizer_settings.all_bounds)
    pareto = []
    for index, model in enumerate(result.models):
        validation = _rollout_metrics(model, batch, validation_slice)
        coefficients = np.asarray(result.coeff_matrices[index], dtype=float)
        try:
            equations = result.symbolic_equations(
                index=index,
                input_features=STATE_NAMES,
                state_names=STATE_NAMES,
                coefficient_threshold=tolerance,
            )
        except (ValueError, IndexError) as exc:
            equations = [f"unavailable: {type(exc).__name__}: {exc}"]
        pareto.append(
            {
                "index": index,
                "objectives": {
                    name: float(value)
                    for name, value in zip(
                        objective_names, result.objective_space[index]
                    )
                },
                "variables": {
                    name: float(value)
                    for name, value in zip(variable_names, result.alphas[index])
                },
                "coefficients": coefficients.tolist(),
                "coefficient_metrics": _coefficient_metrics(
                    coefficients, config, tolerance
                ),
                "validation": validation,
                "symbolic_equations": equations,
            }
        )

    selected = min(
        pareto,
        key=lambda entry: (
            entry["validation"]["rollout_nrmse"],
            entry["coefficient_metrics"]["active_coefficients"],
            entry["objectives"][objective_names[0]],
        ),
    )
    selected_index = selected["index"]
    test_metrics = _rollout_metrics(result.models[selected_index], batch, test_slice)
    config_bytes = config_path.read_bytes()
    elapsed = time.perf_counter() - started
    evaluator = getattr(result.result.algorithm, "evaluator", None)
    actual_evaluations = int(getattr(evaluator, "n_eval", 0))

    return {
        "schema_version": 1,
        "status": "complete",
        "mode": plan.mode,
        "job": {
            **plan.job,
            "population": plan.population,
            "generations": plan.generations,
            "evaluation_budget": plan.evaluation_budget,
        },
        "data": {
            "t_start": float(config["data"]["t_start"]),
            "t_end": plan.t_end,
            "dt": float(config["data"]["dt"]),
            "split_counts": {
                "train": plan.split_counts[0],
                "validation": plan.split_counts[1],
                "test": plan.split_counts[2],
            },
            "initial_conditions": [
                values.tolist() for values in batch.initial_conditions
            ],
        },
        "provenance": {
            **_git_provenance(repo_root),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "versions": _package_versions(),
        },
        "actual_candidate_evaluations": actual_evaluations,
        "failed_candidate_count": int(
            getattr(result.problem, "n_failed_evaluations", 0)
        ),
        "wall_time_seconds": elapsed,
        "pareto": pareto,
        "selected_index": selected_index,
        "selected": selected,
        "test": test_metrics,
    }


def default_output_path(repo_root: Path, plan: RuntimePlan) -> Path:
    suffix = "-smoke" if plan.mode == "smoke" else ""
    return (
        repo_root
        / "benchmarks"
        / "results"
        / "raw"
        / plan.job["experiment"]
        / f"{plan.job['job_id']}{suffix}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--job-id", required=True, help="Job name or zero-based index")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)
    jobs = build_jobs(config, config_path)
    try:
        numeric_index = int(args.job_id)
    except ValueError:
        matches = [job for job in jobs if job["job_id"] == args.job_id]
        if not matches:
            raise SystemExit(f"Unknown job id: {args.job_id}")
        job = matches[0]
    else:
        if not 0 <= numeric_index < len(jobs):
            raise SystemExit(f"Job index must be between 0 and {len(jobs) - 1}")
        job = jobs[numeric_index]

    plan = make_runtime_plan(config, job, smoke=args.smoke)
    output_path = (
        args.output.resolve() if args.output else default_output_path(repo_root, plan)
    )
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Output already exists: {output_path} (use --force to replace it)"
        )
    if config["experiment"]["name"] != "feature_recovery":
        raise SystemExit(
            "run_job.py currently supports only the feature_recovery experiment"
        )

    print(
        f"running {job['job_id']} in {plan.mode} mode: "
        f"{plan.population} x {plan.generations} = {plan.evaluation_budget} evaluations"
    )
    payload = run_feature_recovery(
        config,
        plan,
        config_path=config_path,
        repo_root=repo_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_value(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"completed in {payload['wall_time_seconds']:.2f}s; "
        f"selected validation NRMSE={payload['selected']['validation']['rollout_nrmse']:.6g}; "
        f"test NRMSE={payload['test']['rollout_nrmse']:.6g}"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
