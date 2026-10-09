"""Run one Figure 1 design with the current weak-form weights.

From the repository root, after installing the package:
    python examples/quickstart/goodwin_weak.py --noise 0.10 --seed 11

The defaults are 10% noise and seed 11. A full run uses 2,400 model evaluations
and may take several minutes. The bundled comparison run used earlier weights.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

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
from ddmo.utils.ode_data_generator import ODEDataGenerator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--noise", type=float, default=0.10, choices=[0, 0.05, 0.10, 0.15, 0.20])
parser.add_argument("--seed", type=int, default=11, choices=[11, 22, 33, 44, 55])
args = parser.parse_args()

# The six initial conditions and time grid are the ones used for Figure 1.
initial_conditions = [
    [0.5, 1.0, 1.0],
    [0.2, 1.4, 0.8],
    [0.8, 0.6, 1.5],
    [0.35, 1.2, 1.4],
    [0.7, 1.5, 0.6],
    [1.0, 0.8, 1.2],
]
time = np.arange(0.0, 100.0 + 0.05, 0.1)
goodwin = GoodwinModel(n=9.5)
batch = ODEDataGenerator(
    goodwin.dynamics,
    initial_conditions=initial_conditions,
    time_array=time,
    noise_level=args.noise,
    seed=args.seed,
    clip_min=1e-10,
).generate()

print(f"Generated {len(batch.x_noisy)} trajectories with {len(batch.time)} time points each.")

# Trajectories 1-3 train the models; trajectory 4 selects one Pareto model.
# Trajectories 5-6 are held out until after selection.
data = DataConstructor(
    x_train=batch.x_noisy[:3],
    time_array=batch.time,
    x_dot=None,
    x_0=batch.initial_conditions[:3],
    x_raw=batch.x_noisy[:3],
)


def hill(value, params):
    return 1.0 / (1.0 + np.clip(value, 1e-10, None) ** 9.5)


def hill_name(value, params):
    return f"1/(1+{value}^9.5)"


library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2, include_constant=False, include_interactions=True)
    .add_custom(hill, hill_name)
)
formulation = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=61,
    stride=20,
)
solver = SolverSettings(
    solver_name="pysindy",
    regularization="STLSQ",
    regularization_parameters={"fit_intercept": False, "normalize_columns": False},
    scale_features=True,
)
optimizer = OptimizerSettings(
    library_bounds={},
    formulation_bounds={"degree": (2, 6), "window_length": (21, 151), "stride": (5, 40)},
    solver_bounds={"threshold": (0.001, 1.0), "alpha": (0.000001, 1.0)},
    n_gen=30,
    pop_size=80,
    algorithm="NSGA2",
    seed=args.seed,
    verbose=False,
)

result = identify(
    data_instance=data,
    optimizer_settings=optimizer,
    solver_settings=solver,
    formulation_settings=formulation,
    library_object=library.build,
    objective_functions={
        "trajectory_rmse": trajectory_rmse,
        "durbin_watson_deviation": durbin_watson_deviation,
        "active_coefficients": make_active_coefficient_count(tolerance=1e-6),
    },
)


def rollout_nrmse(model, trajectory_indices):
    """Mean statewise RMSE, scaled by the clean state's standard deviation."""
    clean = batch.x_clean[trajectory_indices]
    starts = batch.initial_conditions[trajectory_indices]
    try:
        predictions = np.asarray([model.simulate(x0, batch.time) for x0 in starts])
    except Exception:
        return np.inf
    rmse = np.sqrt(np.mean((predictions - clean) ** 2, axis=(0, 1)))
    scale = np.std(clean, axis=(0, 1))
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return float(np.mean(rmse / scale))


# Use the same validation rule as the benchmark, including its tie breakers.
choices = []
for index, (model, coefficients, objectives) in enumerate(
    zip(result.models, result.coeff_matrices, result.objective_space)
):
    if not hasattr(model, "simulate") or coefficients is None:
        continue
    if np.asarray(coefficients).ndim != 2 or not np.all(np.isfinite(objectives)):
        continue
    validation_error = rollout_nrmse(model, slice(3, 4))
    if np.isfinite(validation_error):
        active = np.count_nonzero(np.abs(coefficients) > 1e-6)
        choices.append((validation_error, active, objectives[0], index))

validation_error, _, _, selected_index = min(choices)
selected_model = result.models[selected_index]
test_error = rollout_nrmse(selected_model, slice(4, 6))
coefficients = np.asarray(result.coeff_matrices[selected_index])
settings = dict(zip(optimizer.all_bounds, result.alphas[selected_index]))

# The other two Figure 1 panels compare the selected coefficients with truth.
feature_names = list(selected_model.feature_library.get_feature_names(["x", "y", "z"]))
active_mask = getattr(selected_model, "active_mask", None)
if active_mask is not None:
    feature_names = [
        name for name, active in zip(feature_names, active_mask) if active
    ]
truth = np.zeros_like(coefficients)
for row, name in enumerate(feature_names):
    if name == "x":
        truth[row] = [-0.2, 1.0, 0.0]
    elif name == "y":
        truth[row] = [0.0, -0.15, 1.0]
    elif name == "z":
        truth[row] = [0.0, 0.0, -0.1]
    elif name.startswith("1/(1+z^"):
        truth[row] = [1.0, 0.0, 0.0]
coefficient_error = np.linalg.norm(coefficients - truth) / np.linalg.norm(truth)
false_positives = np.count_nonzero((np.abs(coefficients) > 1e-6) & (np.abs(truth) <= 1e-6))

print(f"Weak, optimized | noise={args.noise:.0%} | seed={args.seed}")
print(f"Selected Pareto model: {selected_index}")
print(f"Validation NRMSE: {validation_error:.8g}")
print(f"Test NRMSE:       {test_error:.8g}")
print(f"Coefficient relative error: {coefficient_error:.8g}")
print(f"False-positive coefficients: {false_positives}")
print("Selected settings:")
for name, value in settings.items():
    print(f"  {name}: {float(value):.8g}")

# Compare with the archived, pre-correction paper run.
archive_path = Path(__file__).resolve().parents[2] / "benchmarks/data/historical/noise.zip"
with ZipFile(archive_path) as archive:
    suffix = f"-weak_optimized-noise{round(100 * args.noise):03d}-seed{args.seed:03d}.json"
    saved_name = next(name for name in archive.namelist() if name.endswith(suffix))
    saved = json.loads(archive.read(saved_name))

print("\nHistorical Figure 1 run (earlier weak weights):")
print(f"Validation NRMSE: {saved['selected']['validation']['rollout_nrmse']:.8g}")
print(f"Test NRMSE:       {saved['test']['rollout_nrmse']:.8g}")
print(f"Coefficient relative error: {saved['selected']['coefficient_metrics']['relative_l2_error']:.8g}")
print(f"False-positive coefficients: {saved['selected']['coefficient_metrics']['false_positive_coefficients']}")
print(f"Selected Pareto model: {saved['selected_index']}")
saved_coefficients = np.asarray(saved["selected"]["coefficients"])
print(f"Largest coefficient difference: {np.max(np.abs(coefficients - saved_coefficients)):.8g}")
