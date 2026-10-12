"""Goodwin oscillator weak-form example with generated training data.

This script builds a batch of synthetic Goodwin trajectories instead of loading
the benchmark ``.npy`` files. Edit the configuration block below to control the
noise level, initial conditions, sampling frequency, and simulation time span.

Run from the repository root with:

    PYTHONPATH=src python examples/tutorial_goodwin_generated_batch.py

or after installing the package in editable mode:

    python examples/tutorial_goodwin_generated_batch.py
"""

from __future__ import annotations

import warnings

import numpy as np

from ddmo import (
    FormulationSettings,
    ODEDataGenerator,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.equations.goodwin_eqs import GoodwinModel
from ddmo.library_functions import LibraryBuilder
from ddmo.utils.plotting import plot_ode_data_batch

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data-generation configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NOISE_LEVEL = 0.1
SAMPLING_FREQUENCY = 5.0
T_START = 0.0
T_END = 150.0
INITIAL_CONDITIONS = [
    np.array([0.5, 1.0, 1.0]),
    np.array([0.2, 1.4, 0.8]),
]


goodwin = GoodwinModel(n=9.5)
generator = ODEDataGenerator(
    ode=goodwin.dynamics,
    initial_conditions=INITIAL_CONDITIONS,
    noise_level=NOISE_LEVEL,
    sampling_frequency=SAMPLING_FREQUENCY,
    t_start=T_START,
    t_end=T_END,
    seed=RANDOM_SEED,
    clip_min=1e-8,
)
batch = generator.generate()
data = batch.to_data_constructor()

print("Generated Goodwin training batch:")
print(f"  x_train shape: {data.x_train.shape}")
print(f"  time points: {len(data.time_array)}")
print(f"  noise level: {NOISE_LEVEL}")
print(f"  sampling frequency: {SAMPLING_FREQUENCY} Hz")

library = (
    LibraryBuilder()
    .add_polynomial(max_degree=1, include_interactions=False)
    .add_hill(exponent_param="gamma_1")
    .add_exponential(rate_param="gamma_2")
)

formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=61,
    stride=20,
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

optimizer_settings = OptimizerSettings(
    library_bounds={
        "gamma_1": (5.0, 10.0),
        "gamma_2": (0.0, 5.0),
    },
    formulation_bounds={
        "degree": (2, 5),
        "stride": (5, 20),
        "window_length": (20, 80),
    },
    solver_bounds={
        "threshold": (0.01, 0.2),
        "alpha": (0.01, 0.2),
    },
    n_obj=2,
    n_gen=10,
    pop_size=100,
    algorithm="NSGA3",
    seed=RANDOM_SEED,
    verbose=False,
)

result = identify(
    data_instance=data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
)

print("\nOptimized library/formulation/solver parameters:")
for row_idx, values in enumerate(result.alphas, start=1):
    print(f"\nPareto selection {row_idx}:")
    for name, value in zip(optimizer_settings.all_bounds, values):
        print(f"  {name}: {float(value):.6g}")

if len(result.models) > 0:
    print("\nDiscovered equations for Pareto selection 1:")
    for equation in result.symbolic_equations(
        index=0,
        input_features=["x", "y", "z"],
        state_names=["x", "y", "z"],
        coefficient_threshold=1e-8,
    ):
        print(f"  {equation}")

if len(result.solutions) > 0:
    plot_ode_data_batch(
        batch,
        solution=result.solutions[0],
        variable_names=["x", "y", "z"],
        title="Goodwin oscillator: generated batch, weak-form identification",
    )
else:
    print("No solutions found.")
