"""Goodwin oscillator example using the weak formulation.

This script demonstrates a complete DDMO workflow on the benchmark Goodwin
oscillator data:

1. load benchmark trajectories;
2. construct a candidate library;
3. configure weak-form model identification;
4. optimize library and weak-form parameters;
5. compare the identified trajectory with the observed data.

Run from the repository root with:

    PYTHONPATH=src python examples/quickstart/goodwin_weak.py

or after installing the package in editable mode:

    python examples/quickstart/goodwin_weak.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from ddmo import (
    BENCHMARK_FOLDER,
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.library_functions import LibraryBuilder

warnings.filterwarnings("ignore")


RANDOM_SEED = 42


x_0_train = np.load(BENCHMARK_FOLDER / "x_0.npy")
x_train = np.load(BENCHMARK_FOLDER / "x_train_noise.npy")
time_array = np.load(BENCHMARK_FOLDER / "time.npy")

data = DataConstructor(
    x_train=x_train,
    time_array=time_array,
    x_0=[x_0_train],
    x_raw=x_train,
)

library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2, include_interactions=True)
    .add_hill(exponent_param="gamma_1")
    .add_exponential(rate_param="gamma_2")
)

formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=65,
    stride=50,
)

solver_settings = SolverSettings(
    solver_name="pysindy",
    regularization="STLSQ",
    regularization_parameters={
        "normalize_columns": False,
        "alpha": 0.05,
        "threshold": 0.1,
    },
    scale_features=True,
)

optimizer_settings = OptimizerSettings(
    library_bounds={
        "gamma_1": (8.4, 9.6),
        "gamma_2": (1.0, 10.0),
    },
    formulation_bounds={
        "degree": (2, 7),
        "stride": (50, 250),
        "window_length": (127, 801),
    },
    n_gen=10,
    algorithm="NSGA3",
    pop_size=50,
    seed=RANDOM_SEED,
)


result = identify(
    data_instance=data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
)

print("\nOptimized library/formulation parameters:")
for row_idx, values in enumerate(result.alphas, start=1):
    print(f"\nPareto selection {row_idx}:")
    for name, value in zip(optimizer_settings.all_bounds, values):
        print(f"  {name}: {float(value):.6g}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(result.solutions[0], label="identified trajectory")
ax.plot(x_train, alpha=0.5, linestyle="--", label="observed noisy data")
ax.set_title("Goodwin oscillator: weak-form DDMO identification")
ax.set_xlabel("time index")
ax.set_ylabel("state value")
ax.legend()
fig.tight_layout()
plt.show()
