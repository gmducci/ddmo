"""Reproduce one Figure 2 joint-library identification run.

Run from the repository root after installing ddmo:

    python examples/quickstart/goodwin_joint.py --block 4

The full run evaluates 2,400 candidates. Use --smoke for a short API walkthrough
that does not compare with the saved publication result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tomllib
from scipy.signal import savgol_filter

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
    trajectory_rmse,
)
from ddmo.utils.ode_data_generator import ODEDataBatch, ODEDataGenerator

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "benchmarks/joint/config.toml"


def saved_run(block: int, path: Path | None) -> dict:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if block == 4:
        reference = ROOT / "benchmarks/data/figure2_tutorial_block04.json"
        return json.loads(reference.read_text(encoding="utf-8"))
    raise ValueError("For blocks other than 4, pass --saved-run with that block's result JSON")


def rollout_nrmse(model, clean: np.ndarray, initial_conditions, time: np.ndarray) -> float:
    """Mean statewise RMSE scaled by clean-state standard deviations."""
    try:
        predicted = np.asarray([model.simulate(x0, time) for x0 in initial_conditions])
    except Exception:
        return float("inf")
    rmse = np.sqrt(np.mean((predicted - clean) ** 2, axis=(0, 1)))
    scale = np.std(clean, axis=(0, 1))
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return float(np.mean(rmse / scale))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, choices=range(1, 11), default=4,
                        help="Figure 2 repetition / Latin-hypercube initial-condition block")
    parser.add_argument("--smoke", action="store_true", help="use 12 x 3 evaluations and t=20")
    parser.add_argument("--saved-run", type=Path,
                        help="compare with a specific result JSON (required for full runs other than block 4)")
    args = parser.parse_args()

    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    n_train = int(config["data"]["train_trajectories"])
    n_validation = int(config["data"]["validation_trajectories"])
    n_test = 1 if args.smoke else int(config["data"]["test_trajectories"])
    run_number = args.block - 1
    data_seed = int(config["experiment"]["seeds"][run_number])
    optimizer_seed = int(config["experiment"]["optimizer_seeds"][run_number])
    budget = config["smoke"] if args.smoke else config["optimizer"]
    population = int(budget["population"])
    generations = int(budget["generations"])
    t_end = float(budget["t_end"] if args.smoke else config["data"]["t_end"])

    # The third designed training start is generated to preserve the exact
    # random-noise stream, then omitted from the Figure 2 two-train split.
    with (CONFIG.parent / config["data"]["initial_condition_design"]).open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = {row["role"]: row for row in csv.DictReader(stream)
                if int(row["block"]) == args.block}
    roles = ["train_1", "train_2", "train_3", "validation", "test_1", "test_2"]
    initial_conditions = [
        np.array([float(rows[role][name]) for name in ("x0", "y0", "z0")])
        for role in roles
    ]
    truth = config["truth"]
    goodwin = GoodwinModel(
        k1=float(truth["k1"]), k2=float(truth["k2"]),
        k3=float(truth["k3"]), k4=float(truth["k4"]),
        k5=float(truth["k5"]), k6=float(truth["k6"]),
        K1=float(truth["K1"]), n=float(truth["hill_exponent"]),
    )
    dt = float(config["data"]["dt"])
    time = np.arange(float(config["data"]["t_start"]), t_end + 0.5 * dt, dt)
    generated = ODEDataGenerator(
        goodwin.dynamics, initial_conditions=initial_conditions,
        time_array=time, noise_level=float(config["experiment"]["noise_levels"][0]),
        seed=data_seed, clip_min=1e-10,
    ).generate()
    indices = [0, 1, 3, 4] if args.smoke else [0, 1, 3, 4, 5]
    batch = ODEDataBatch(
        time=generated.time,
        x_clean=generated.x_clean[indices],
        x_noisy=generated.x_noisy[indices],
        x_dot_clean=generated.x_dot_clean[indices],
        initial_conditions=[generated.initial_conditions[index] for index in indices],
    )
    validation = slice(n_train, n_train + n_validation)
    test = slice(n_train + n_validation, n_train + n_validation + n_test)

    # Replace this block with observed trajectories and their initial conditions
    # to adapt the workflow. Figure 2 uses two training trajectories at 10% noise.
    observations = batch.x_noisy[:n_train]
    sg = config["strong_derivative"]
    preprocessed = np.clip(
        savgol_filter(observations, window_length=int(sg["window_length"]),
                      polyorder=int(sg["polynomial_order"]), axis=1),
        1e-10, None,
    )
    data = DataConstructor(
        x_train=preprocessed,
        x_raw=preprocessed,
        time_array=batch.time,
        x_0=batch.initial_conditions[:n_train],
    )

    # The two nonlinear families compete with degree-2 polynomial terms.
    def hill(value, parameters):
        return 1.0 / (1.0 + np.clip(value, 1e-10, None) ** parameters["gamma_1"])

    def hill_name(value, parameters):
        return f"1/(1+{value}^{parameters['gamma_1']:.6g})"

    library = (
        LibraryBuilder()
        .add_polynomial(max_degree=2, include_constant=False, include_interactions=True)
        .add_custom(hill, hill_name)
        .add_exponential("gamma_2")
    )
    weak = config["weak_initial"]
    formulation = FormulationSettings(
        formulation="weak", test_function=weak["family"],
        degree=int(weak["degree"]), window_length=int(weak["window_length"]),
        stride=int(weak["stride"]),
    )
    solver = SolverSettings(
        solver_name="pysindy", regularization="STLSQ",
        regularization_parameters={"fit_intercept": False, "normalize_columns": False},
        scale_features=True,
    )
    bounds = config["bounds"]
    optimizer = OptimizerSettings(
        library_bounds={key: tuple(value) for key, value in bounds["library"].items()},
        formulation_bounds={key: tuple(value) for key, value in bounds["formulation"].items()},
        solver_bounds={key: tuple(value) for key, value in bounds["solver"].items()},
        n_gen=generations, pop_size=population,
        algorithm="NSGA2", seed=optimizer_seed, verbose=False,
    )
    result = identify(
        data_instance=data, library_object=library.build,
        formulation_settings=formulation, solver_settings=solver,
        optimizer_settings=optimizer,
        objective_functions={
            "trajectory_rmse": trajectory_rmse,
            "durbin_watson_deviation": durbin_watson_deviation,
            "active_coefficients": make_active_coefficient_count(tolerance=1e-6),
        },
    )

    # Select by the clean validation trajectory, as in the synthetic benchmark.
    # Its two test trajectories are untouched until selection is complete.
    choices = []
    for index, (model, coefficients, objectives) in enumerate(
        zip(result.models, result.coeff_matrices, result.objective_space)
    ):
        if not hasattr(model, "simulate") or coefficients is None:
            continue
        if np.asarray(coefficients).ndim != 2 or not np.all(np.isfinite(objectives)):
            continue
        validation_error = rollout_nrmse(
            model, batch.x_clean[validation], batch.initial_conditions[validation], batch.time
        )
        if np.isfinite(validation_error):
            active = int(np.count_nonzero(np.abs(coefficients) > 1e-6))
            choices.append((validation_error, active, float(objectives[0]), index))
    if not choices:
        raise RuntimeError("No Pareto model completed the validation rollout")
    validation_error, _, _, index = min(choices)
    test_error = rollout_nrmse(
        result.models[index], batch.x_clean[test], batch.initial_conditions[test], batch.time
    )
    coefficients = np.asarray(result.coeff_matrices[index], dtype=float)
    settings = dict(zip(optimizer.all_bounds, result.alphas[index]))

    print(f"Figure 2 repetition {args.block} ({'smoke' if args.smoke else 'production'}; "
          f"{population * generations} evaluations)")
    print(f"Selected Pareto model: {index}")
    print(f"Validation NRMSE: {validation_error:.8g}")
    print(f"Test NRMSE:       {test_error:.8g}")
    print("Selected settings:")
    for name, value in settings.items():
        print(f"  {name}: {float(value):.8g}")
    print("Identified equations:")
    for equation in result.symbolic_equations(
        index=index, input_features=["x", "y", "z"], state_names=["x", "y", "z"],
        coefficient_threshold=1e-6,
    ):
        print(f"  {equation}")

    if not args.smoke:
        saved = saved_run(args.block, args.saved_run)
        if int(saved["job"]["ic_block"]) != args.block:
            raise ValueError("The saved result belongs to another block")
        expected = saved["selected"]
        np.testing.assert_allclose(coefficients, expected["coefficients"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(test_error, saved["test"]["rollout_nrmse"], rtol=1e-9)
        np.testing.assert_allclose(validation_error, expected["validation"]["rollout_nrmse"], rtol=1e-9)
        if index != saved["selected_index"]:
            raise AssertionError("Selected Pareto index differs from the saved run")
        print("Matches the saved Figure 2 run.")


if __name__ == "__main__":
    main()
