"""Shared execution machinery for the Goodwin benchmark figures."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from benchmarks.common.plan_runs import build_jobs, load_config
from ddmo import (
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.equations.goodwin_eqs import GoodwinModel
from ddmo.library_functions import LibraryBuilder
from ddmo.optimizer import (
    durbin_watson_deviation,
    make_active_coefficient_count,
    trajectory_mse,
    trajectory_rmse,
)
from ddmo.utils.ode_data_generator import ODEDataBatch, ODEDataGenerator

STATE_NAMES = ["x", "y", "z"]


@dataclass(frozen=True)
class GoodwinRuntimePlan:
    """Resolved production or smoke settings for one Goodwin job."""

    job: dict[str, Any]
    population: int
    generations: int
    t_end: float
    split_counts: tuple[int, int, int]
    mode: str

    @property
    def evaluation_budget(self) -> int:
        return self.population * self.generations


def make_runtime_plan(
    config: dict[str, Any], job: dict[str, Any], *, smoke: bool
) -> GoodwinRuntimePlan:
    """Apply explicit smoke overrides without mutating the frozen config."""
    data = config["data"]
    if smoke:
        smoke_config = config.get("smoke", {})
        return GoodwinRuntimePlan(
            job=job,
            population=int(smoke_config.get("population", 12)),
            generations=int(smoke_config.get("generations", 3)),
            t_end=float(smoke_config.get("t_end", 20.0)),
            split_counts=(int(job.get("training_trajectories", 1)), 1, 1),
            mode="smoke",
        )
    return GoodwinRuntimePlan(
        job=job,
        population=int(config["optimizer"]["population"]),
        generations=int(config["optimizer"]["generations"]),
        t_end=float(data["t_end"]),
        split_counts=(
            int(job.get("training_trajectories", data["train_trajectories"])),
            int(data["validation_trajectories"]),
            int(data["test_trajectories"]),
        ),
        mode="production",
    )


def resolve_job(config_path: Path, job_id: str) -> tuple[dict, dict]:
    """Load a config and resolve a named or zero-based job identifier."""
    config = load_config(config_path)
    jobs = build_jobs(config, config_path)
    try:
        numeric_index = int(job_id)
    except ValueError:
        matches = [job for job in jobs if job["job_id"] == job_id]
        if not matches:
            raise ValueError(f"Unknown job id: {job_id}")
        job = matches[0]
    else:
        if not 0 <= numeric_index < len(jobs):
            raise ValueError(f"Job index must be between 0 and {len(jobs) - 1}")
        job = jobs[numeric_index]
    return config, job


def default_output_paths(
    repo_root: Path, plan: GoodwinRuntimePlan
) -> tuple[Path, Path]:
    suffix = "-smoke" if plan.mode == "smoke" else ""
    stem = (
        repo_root
        / ".local"
        / "benchmark_runs"
        / plan.job["experiment"]
        / "raw"
        / f"{plan.job['job_id']}{suffix}"
    )
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def generate_batch(config: dict, plan: GoodwinRuntimePlan) -> ODEDataBatch:
    """Generate all frozen train, validation, and test trajectories."""
    data = config["data"]
    truth = config["truth"]
    required = sum(plan.split_counts)
    configured = data.get("initial_conditions", [])
    design_path = data.get("initial_condition_design")
    selected_indices = None
    if design_path:
        source = Path(plan.job["config"]).resolve().parent / str(design_path)
        roles = ["train_1", "train_2", "train_3", "validation", "test_1", "test_2"]
        with source.open(newline="", encoding="utf-8") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if int(row["block"]) == int(plan.job["ic_block"])
            ]
        by_role = {row["role"]: row for row in rows}
        if set(by_role) != set(roles):
            raise ValueError(f"Invalid initial-condition block {plan.job['ic_block']}")
        configured = [
            [float(by_role[role][name]) for name in ("x0", "y0", "z0")]
            for role in roles
        ]
        n_train = plan.split_counts[0]
        selected_indices = list(range(n_train)) + [3, 4, 5]
        required = 6
    if len(configured) < required:
        raise ValueError(
            f"data.initial_conditions needs at least {required} entries; "
            f"found {len(configured)}"
        )
    initial_conditions = [
        np.asarray(values, dtype=float) for values in configured[:required]
    ]
    system = GoodwinModel(
        k1=float(truth.get("k1", 1.0)),
        k2=float(truth.get("k2", 0.2)),
        k3=float(truth.get("k3", 1.0)),
        k4=float(truth.get("k4", 0.15)),
        k5=float(truth.get("k5", 1.0)),
        k6=float(truth.get("k6", 0.1)),
        K1=float(truth.get("K1", 1.0)),
        n=float(truth["hill_exponent"]),
    )
    dt = float(data["dt"])
    time_array = np.arange(float(data["t_start"]), plan.t_end + 0.5 * dt, dt)
    batch = ODEDataGenerator(
        system.dynamics,
        initial_conditions=initial_conditions,
        time_array=time_array,
        noise_level=float(plan.job["noise_level"]),
        seed=int(plan.job["data_seed"]),
        clip_min=1e-10,
    ).generate()
    if selected_indices is None:
        return batch
    return ODEDataBatch(
        time=batch.time,
        x_clean=batch.x_clean[selected_indices],
        x_noisy=batch.x_noisy[selected_indices],
        x_dot_clean=batch.x_dot_clean[selected_indices],
        initial_conditions=[
            batch.initial_conditions[index] for index in selected_indices
        ],
    )


def split_slices(
    counts: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    n_train, n_validation, n_test = counts
    validation_end = n_train + n_validation
    return (
        slice(0, n_train),
        slice(n_train, validation_end),
        slice(validation_end, validation_end + n_test),
    )


def build_library(
    config: dict[str, Any], *, adaptive_exponent: bool
) -> tuple[LibraryBuilder, dict[str, tuple[float, float]]]:
    """Build the configured Goodwin candidate library plus Hill inhibition."""
    library_config = config.get("library", {})
    polynomial_degree = int(library_config.get("polynomial_degree", 1))
    if polynomial_degree > 1:
        library = LibraryBuilder().add_polynomial(
            max_degree=polynomial_degree,
            include_constant=False,
            include_interactions=bool(library_config.get("include_interactions", True)),
        )
    else:
        library = LibraryBuilder().add_identity()
    if adaptive_exponent:

        def adaptive_hill(value, params):
            positive_value = np.clip(value, 1e-10, None)
            return 1.0 / (1.0 + positive_value ** params["gamma_1"])

        def adaptive_hill_name(value, params):
            return f"1/(1+{value}^{params['gamma_1']:.6g})"

        library.add_custom(adaptive_hill, adaptive_hill_name)
        exponential_param = library_config.get("exponential_rate_param")
        if exponential_param:
            library.add_exponential(str(exponential_param))
        library_bounds = {
            name: tuple(bounds) for name, bounds in config["bounds"]["library"].items()
        }
    else:
        exponent = float(config["truth"]["hill_exponent"])

        def hill(value, params):
            del params
            positive_value = np.clip(value, 1e-10, None)
            return 1.0 / (1.0 + positive_value**exponent)

        def hill_name(value, params):
            del params
            return f"1/(1+{value}^{exponent:.6g})"

        library.add_custom(hill, hill_name)
        library_bounds = {}
    return library, library_bounds


def _savgol_states(config: dict[str, Any], noisy: np.ndarray) -> np.ndarray:
    """Apply the benchmark's shared Savitzky-Golay state preprocessing."""
    settings = config["strong_derivative"]
    if settings["method"] != "savitzky_golay":
        raise ValueError(f"Unsupported strong derivative method: {settings['method']}")
    n_time = noisy.shape[1]
    window_length = int(settings["window_length"])
    if window_length > n_time:
        window_length = n_time if n_time % 2 else n_time - 1
    polynomial_order = min(int(settings["polynomial_order"]), window_length - 1)
    smoothed = savgol_filter(
        noisy,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=0,
        axis=1,
    )
    # Goodwin concentrations are positive and the candidate Hill term uses a
    # fractional exponent. Savitzky-Golay edge overshoot can otherwise create
    # small negative values and undefined fractional powers in the library.
    return np.clip(smoothed, 1e-10, None)


def _strong_training_data(
    config: dict[str, Any], batch: ODEDataBatch, train_slice: slice
) -> DataConstructor:
    settings = config["strong_derivative"]
    noisy = batch.x_noisy[train_slice]
    smoothed = _savgol_states(config, noisy)
    window_length = min(int(settings["window_length"]), noisy.shape[1])
    if window_length % 2 == 0:
        window_length -= 1
    polynomial_order = min(int(settings["polynomial_order"]), window_length - 1)
    dt = float(batch.time[1] - batch.time[0])
    derivatives = savgol_filter(
        noisy,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=1,
        delta=dt,
        axis=1,
    )
    return DataConstructor(
        x_train=smoothed,
        time_array=batch.time,
        x_dot=derivatives,
        x_0=batch.initial_conditions[train_slice],
        x_raw=noisy,
    )


def _weak_training_data(
    config: dict[str, Any],
    batch: ODEDataBatch,
    train_slice: slice,
    *,
    savgol_preprocess: bool,
) -> DataConstructor:
    observations = batch.x_noisy[train_slice]
    if savgol_preprocess:
        observations = _savgol_states(config, observations)
    return DataConstructor(
        x_train=observations,
        time_array=batch.time,
        x_dot=None,
        x_0=batch.initial_conditions[train_slice],
        x_raw=observations,
    )


def build_identification_settings(
    config: dict[str, Any],
    plan: GoodwinRuntimePlan,
    batch: ODEDataBatch,
    train_slice: slice,
    *,
    adaptive_exponent: bool,
) -> tuple[
    DataConstructor,
    LibraryBuilder,
    OptimizerSettings,
    SolverSettings,
    FormulationSettings,
]:
    """Map a configured Goodwin method to the public ddmo API."""
    method = plan.job["method"]
    library, library_bounds = build_library(config, adaptive_exponent=adaptive_exponent)
    formulation_bounds: dict[str, tuple[float, float]] = {}
    if method == "strong":
        data_instance = _strong_training_data(config, batch, train_slice)
        formulation_settings = FormulationSettings(formulation="strong")
    elif method in {
        "weak_fixed",
        "weak_optimized",
        "weak_optimized_sg",
        "weak_joint",
        "weak_joint_sg",
    }:
        data_instance = _weak_training_data(
            config,
            batch,
            train_slice,
            savgol_preprocess=method in {"weak_optimized_sg", "weak_joint_sg"},
        )
        base = (
            config["weak_fixed"] if "weak_fixed" in config else config["weak_initial"]
        )
        formulation_settings = FormulationSettings(
            formulation="weak",
            test_function=base["family"],
            degree=int(base["degree"]),
            window_length=int(base["window_length"]),
            stride=int(base["stride"]),
        )
        if method in {
            "weak_optimized",
            "weak_optimized_sg",
            "weak_joint",
            "weak_joint_sg",
        }:
            formulation_bounds = {
                name: tuple(bounds)
                for name, bounds in config["bounds"]["formulation"].items()
            }
            if plan.mode == "smoke" and "window_length" in formulation_bounds:
                lower, upper = formulation_bounds["window_length"]
                largest_window = len(batch.time) - 2
                if largest_window % 2 == 0:
                    largest_window -= 1
                formulation_bounds["window_length"] = (
                    lower,
                    min(upper, float(largest_window)),
                )
    else:
        raise ValueError(f"Unsupported Goodwin method: {method}")

    solver_bounds = {
        name: tuple(bounds) for name, bounds in config["bounds"]["solver"].items()
    }
    optimizer_settings = OptimizerSettings(
        library_bounds=library_bounds,
        formulation_bounds=formulation_bounds,
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
    return (
        data_instance,
        library,
        optimizer_settings,
        solver_settings,
        formulation_settings,
    )


def _objectives(config: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    objectives = {}
    for name in config["optimizer"]["objectives"]:
        if name == "trajectory_mse":
            objectives[name] = trajectory_mse
        elif name == "trajectory_rmse":
            objectives[name] = trajectory_rmse
        elif name in {"durbin_watson_deviation", "dw_coefficient"}:
            objectives[name] = durbin_watson_deviation
        elif name in {"active_coefficients", "active_coefficient_count"}:
            objectives[name] = make_active_coefficient_count(tolerance=tolerance)
        else:
            raise ValueError(f"Unsupported Goodwin objective: {name}")
    return objectives


def rollout(
    model: Any, batch: ODEDataBatch, split: slice
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Simulate one fitted model and calculate clean-reference NRMSE."""
    references = batch.x_clean[split]
    initial_conditions = batch.initial_conditions[split]
    try:
        predictions = np.asarray(
            [
                model.simulate(initial_condition, batch.time)
                for initial_condition in initial_conditions
            ]
        )
    except Exception as exc:
        return (
            {
                "rollout_nrmse": math.inf,
                "per_state_nrmse": [math.inf] * references.shape[-1],
                "simulation_error": f"{type(exc).__name__}: {exc}",
            },
            None,
        )
    axes = (0, 1)
    rmse = np.sqrt(np.mean((predictions - references) ** 2, axis=axes))
    scale = np.std(references, axis=axes)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    per_state = rmse / scale
    return (
        {
            "rollout_nrmse": float(np.mean(per_state)),
            "per_state_nrmse": per_state.tolist(),
            "simulation_error": None,
        },
        predictions,
    )


def _feature_names(model: Any) -> list[str]:
    names = list(model.feature_library.get_feature_names(STATE_NAMES))
    active_mask = getattr(model, "active_mask", None)
    if active_mask is not None:
        names = [
            name
            for name, active in zip(names, np.asarray(active_mask, dtype=bool))
            if active
        ]
    return names


def true_coefficient_matrix(
    feature_names: list[str], config: dict[str, Any]
) -> np.ndarray:
    """Build Goodwin truth coefficients in the fitted feature order."""
    truth = config["truth"]
    matrix = np.zeros((len(feature_names), 3), dtype=float)
    for index, name in enumerate(feature_names):
        if name == "x":
            matrix[index] = [
                -float(truth.get("k2", 0.2)),
                float(truth.get("k3", 1.0)),
                0.0,
            ]
        elif name == "y":
            matrix[index] = [
                0.0,
                -float(truth.get("k4", 0.15)),
                float(truth.get("k5", 1.0)),
            ]
        elif name == "z":
            matrix[index] = [0.0, 0.0, -float(truth.get("k6", 0.1))]
        elif name.startswith("1/(1+z^"):
            matrix[index] = [float(truth.get("k1", 1.0)), 0.0, 0.0]
    return matrix


def coefficient_metrics(
    coefficients: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    tolerance = float(config["selection"]["coefficient_zero_tolerance"])
    estimated = np.asarray(coefficients, dtype=float)
    truth = true_coefficient_matrix(feature_names, config)
    if estimated.shape != truth.shape:
        return {
            "relative_l2_error": math.inf,
            "support_accuracy": 0.0,
            "support_precision": 0.0,
            "support_recall": 0.0,
            "support_f1": 0.0,
            "support_exact": False,
            "false_positive_coefficients": int(estimated.size),
            "false_negative_coefficients": int(np.count_nonzero(truth)),
            "active_coefficients": int(np.count_nonzero(np.abs(estimated) > tolerance)),
        }
    truth_support = np.abs(truth) > tolerance
    estimated_support = np.abs(estimated) > tolerance
    true_positives = int(np.count_nonzero(estimated_support & truth_support))
    false_positives = int(np.count_nonzero(estimated_support & ~truth_support))
    false_negatives = int(np.count_nonzero(~estimated_support & truth_support))
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "relative_l2_error": float(
            np.linalg.norm(estimated - truth) / np.linalg.norm(truth)
        ),
        "support_accuracy": float(np.mean(estimated_support == truth_support)),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "support_exact": bool(np.array_equal(estimated_support, truth_support)),
        "false_positive_coefficients": false_positives,
        "false_negative_coefficients": false_negatives,
        "active_coefficients": int(np.count_nonzero(estimated_support)),
    }


def execute_goodwin_job(
    config: dict[str, Any],
    plan: GoodwinRuntimePlan,
    *,
    config_path: Path,
    repo_root: Path,
    adaptive_exponent: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute one Goodwin job and return JSON metadata plus trajectory arrays."""
    started = time.perf_counter()
    batch = generate_batch(config, plan)
    train_slice, validation_slice, test_slice = split_slices(plan.split_counts)
    (
        data_instance,
        library,
        optimizer_settings,
        solver_settings,
        formulation_settings,
    ) = build_identification_settings(
        config,
        plan,
        batch,
        train_slice,
        adaptive_exponent=adaptive_exponent,
    )
    objectives = _objectives(config)
    result = identify(
        data_instance=data_instance,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings,
        library_object=library.build,
        objective_functions=objectives,
    )
    valid_indices = [
        index
        for index, (model, coefficients, objective_values) in enumerate(
            zip(result.models, result.coeff_matrices, result.objective_space)
        )
        if hasattr(model, "simulate")
        and coefficients is not None
        and np.asarray(coefficients).ndim == 2
        and np.all(np.isfinite(objective_values))
    ]
    if not valid_indices:
        failed = int(getattr(result.problem, "n_failed_evaluations", 0))
        raise RuntimeError(
            "Optimization produced no valid fitted models "
            f"({failed} failed candidate evaluations)."
        )

    objective_names = list(objectives)
    variable_names = list(optimizer_settings.all_bounds)
    pareto = []
    validation_predictions: dict[int, np.ndarray] = {}
    for index in valid_indices:
        model = result.models[index]
        validation, predictions = rollout(model, batch, validation_slice)
        if predictions is not None:
            validation_predictions[index] = predictions
        coefficients = np.asarray(result.coeff_matrices[index], dtype=float)
        feature_names = _feature_names(model)
        metrics = coefficient_metrics(coefficients, feature_names, config)
        try:
            equations = result.symbolic_equations(
                index=index,
                input_features=STATE_NAMES,
                state_names=STATE_NAMES,
                coefficient_threshold=float(
                    config["selection"]["coefficient_zero_tolerance"]
                ),
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
                "feature_names": feature_names,
                "coefficients": coefficients.tolist(),
                "coefficient_metrics": metrics,
                "validation": validation,
                "symbolic_equations": equations,
            }
        )

    selectable = [
        entry
        for entry in pareto
        if math.isfinite(float(entry["validation"]["rollout_nrmse"]))
    ]
    if not selectable:
        raise RuntimeError("No Pareto model completed the validation rollout.")
    selected = min(
        selectable,
        key=lambda entry: (
            entry["validation"]["rollout_nrmse"],
            entry["coefficient_metrics"]["active_coefficients"],
            entry["objectives"][objective_names[0]],
        ),
    )
    selected_index = int(selected["index"])
    selected_model = result.models[selected_index]
    test_metrics, test_predictions = rollout(selected_model, batch, test_slice)
    train_metrics, train_predictions = rollout(selected_model, batch, train_slice)
    if test_predictions is None:
        test_predictions = np.full_like(batch.x_clean[test_slice], np.nan)
    if train_predictions is None:
        train_predictions = np.full_like(batch.x_clean[train_slice], np.nan)

    config_bytes = config_path.read_bytes()
    evaluator = getattr(result.result.algorithm, "evaluator", None)
    actual_evaluations = int(getattr(evaluator, "n_eval", 0))
    elapsed = time.perf_counter() - started
    exponent_metrics = None
    library_parameter_recovery = None
    if adaptive_exponent:
        recovered = float(selected["variables"]["gamma_1"])
        true_exponent = float(config["truth"]["hill_exponent"])
        exponent_metrics = {
            "truth": true_exponent,
            "recovered": recovered,
            "absolute_error": abs(recovered - true_exponent),
            "relative_error": abs(recovered - true_exponent) / abs(true_exponent),
        }
        library_parameter_recovery = {
            "gamma_1": exponent_metrics,
            "gamma_2": {
                "truth": None,
                "recovered": float(selected["variables"]["gamma_2"]),
                "identifiable_from_ground_truth": False,
            }
            if "gamma_2" in selected["variables"]
            else None,
        }

    payload = {
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
            **git_provenance(repo_root),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "versions": package_versions(),
        },
        "actual_candidate_evaluations": actual_evaluations,
        "failed_candidate_count": int(
            getattr(result.problem, "n_failed_evaluations", 0)
        ),
        "excluded_invalid_result_count": len(result.models) - len(valid_indices),
        "wall_time_seconds": elapsed,
        "pareto": pareto,
        "selected_index": selected_index,
        "selected": selected,
        "train": train_metrics,
        "test": test_metrics,
        "exponent_recovery": exponent_metrics,
        "library_parameter_recovery": library_parameter_recovery,
    }
    artifacts = {
        "time": batch.time,
        "initial_conditions": np.asarray(batch.initial_conditions),
        "train_clean": batch.x_clean[train_slice],
        "train_noisy": batch.x_noisy[train_slice],
        "train_preprocessed": np.asarray(data_instance.x_train),
        "train_prediction": train_predictions,
        "validation_clean": batch.x_clean[validation_slice],
        "validation_noisy": batch.x_noisy[validation_slice],
        "validation_prediction": validation_predictions[selected_index],
        "test_clean": batch.x_clean[test_slice],
        "test_noisy": batch.x_noisy[test_slice],
        "test_prediction": test_predictions,
    }
    return payload, artifacts


def git_provenance(repo_root: Path) -> dict[str, Any]:
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


def package_versions() -> dict[str, str]:
    packages = ["ddmo", "numpy", "scipy", "scikit-learn", "pysindy", "pymoo"]
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def json_value(value: Any) -> Any:
    """Convert NumPy and non-finite values into strict JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_outputs(
    json_path: Path,
    artifact_path: Path,
    payload: dict[str, Any],
    artifacts: dict[str, np.ndarray],
) -> None:
    """Write strict JSON metadata and compressed trajectory arrays."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **artifacts)
    payload["trajectory_artifact"] = artifact_path.name
    json_path.write_text(
        json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
