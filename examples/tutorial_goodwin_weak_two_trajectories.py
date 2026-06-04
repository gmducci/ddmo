"""Goodwin oscillator example with two weak-form trajectories.

This tutorial trains DDMO on two synthetic Goodwin oscillator trajectories with
different initial conditions. The regression uses the weak formulation, which
assembles windowed integral equations from each trajectory independently and
then stacks the resulting regression rows.

Run from the repository root with:

    PYTHONPATH=src python examples/tutorial_goodwin_weak_two_trajectories.py

or after installing the package in editable mode:

    python examples/tutorial_goodwin_weak_two_trajectories.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from ddmo import DataConstructor, FormulationSettings, OptimizerSettings, SolverSettings
from ddmo import identify
from ddmo.equations.goodwin_eqs import GoodwinModel
from ddmo.library_functions import LibraryBuilder

warnings.filterwarnings("ignore")

RANDOM_SEED = 42


def generate_data(
    *,
    noise_level: float = 0.02,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Generate two noisy Goodwin trajectories and exact clean derivatives."""
    rng = np.random.default_rng(seed)
    system = GoodwinModel(n=9.5)
    time = np.linspace(0.0, 120.0, 2000)
    initial_conditions = [
        np.array([0.5, 1.0, 1.0]),
        np.array([0.2, 1.4, 0.8]),
    ]

    observed_trajectories = []
    derivative_trajectories = []

    for x0 in initial_conditions:
        x_clean = odeint(system.dynamics, x0, time)
        x_dot = np.asarray(
            [system.dynamics(state, t) for state, t in zip(x_clean, time)]
        )

        if noise_level > 0:
            scale = np.std(x_clean, axis=0)
            noise = rng.normal(loc=0.0, scale=noise_level, size=x_clean.shape) * scale
            x_observed = x_clean + noise
            # The Hill candidate uses fractional powers during optimization.
            # Keep noisy states positive to avoid invalid negative bases.
            x_observed = np.clip(x_observed, 1e-8, None)
        else:
            x_observed = x_clean.copy()

        observed_trajectories.append(x_observed)
        derivative_trajectories.append(x_dot)

    return (
        time,
        np.stack(observed_trajectories, axis=0),
        np.stack(derivative_trajectories, axis=0),
        initial_conditions,
    )


def split_solution(
    solution: np.ndarray | None,
    n_trajectories: int,
    n_time: int,
) -> list[np.ndarray | None]:
    """Split the solver's stacked output into per-trajectory arrays."""
    if solution is None:
        return [None] * n_trajectories
    return [solution[i * n_time : (i + 1) * n_time] for i in range(n_trajectories)]


def make_plot(
    time: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray | None,
) -> None:
    """Plot observed and identified trajectories for x, y, and z."""
    variable_names = ["x", "y", "z"]
    predicted_blocks = split_solution(predicted, observed.shape[0], len(time))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for var_idx, name in enumerate(variable_names):
        ax = axes[var_idx]
        for traj_idx in range(observed.shape[0]):
            ax.plot(
                time,
                observed[traj_idx, :, var_idx],
                alpha=0.55,
                label=f"observed {name}, traj {traj_idx + 1}",
            )

            predicted_traj = predicted_blocks[traj_idx]
            if predicted_traj is not None:
                n = min(len(time), len(predicted_traj))
                ax.plot(
                    time[:n],
                    predicted_traj[:n, var_idx],
                    linestyle="--",
                    label=f"identified {name}, traj {traj_idx + 1}",
                )

        ax.set_ylabel(name)
        ax.legend()

    axes[-1].set_xlabel("time")
    fig.suptitle("Goodwin oscillator weak-form identification, two trajectories")
    fig.tight_layout()
    plt.show()


time, x_train, x_dot, initial_conditions = generate_data(noise_level=0.1)

library = (
    LibraryBuilder()
    .add_polynomial(max_degree=1, include_interactions=False)
    .add_hill(exponent_param="gamma_1")
    .add_exponential(rate_param="gamma_2")
)

data = DataConstructor(
    x_train=x_train,
    time_array=time,
    x_dot=x_dot,
    x_0=initial_conditions,
    x_raw=x_train,
)

optimizer_settings = OptimizerSettings(
    library_bounds={
        "gamma_1": (5.0, 10.0),
        "gamma_2": (0.0, 5.0),
    },
    formulation_bounds={
        "degree": (2, 5),
        "stride": (10, 50),
        "window_length": (40, 140),
    },
    solver_bounds={
        "threshold": (0.001, 0.2),
        "alpha": (0.0001, 0.1),
    },
    n_obj=2,
    n_gen=10,
    pop_size=40,
    algorithm="NSGA3",
    seed=RANDOM_SEED,
    verbose=True,
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

formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=81,
    stride=25,
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

if len(result.solutions) > 0:
    make_plot(time, x_train, result.solutions[0])
else:
    print("No solutions found.")

print(result.coeff_matrices)
