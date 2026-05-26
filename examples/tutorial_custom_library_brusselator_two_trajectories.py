"""Custom-library Brusselator example with two weak-form trajectories.

This tutorial extends the custom-library Brusselator example to train on two
synthetic trajectories with different initial conditions. The model uses the
weak formulation, so the regression problem is assembled from windowed
integrals rather than pointwise derivative estimates.

Run from the repository root with:

    PYTHONPATH=src python examples/tutorial_custom_library_brusselator_two_trajectories.py

or after installing the package in editable mode:

    python examples/tutorial_custom_library_brusselator_two_trajectories.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from ddmo import DataConstructor, FormulationSettings, OptimizerSettings, SolverSettings
from ddmo import identify
from ddmo.equations.brussellator import Brussellator
from ddmo.library_functions import LibraryBuilder

warnings.filterwarnings("ignore")


def build_custom_library() -> LibraryBuilder:
    """Create a candidate library with a parameterized interaction term."""

    def powered_interaction(x: np.ndarray, y: np.ndarray, params: dict[str, float]):
        g1 = params["g1"]
        g2 = params["g2"]
        return (x**g1) * (y**g2)

    def powered_interaction_name(x: str, y: str, params: dict[str, float]) -> str:
        g1 = params["g1"]
        g2 = params["g2"]
        return f"{x}^{g1:.2f}*{y}^{g2:.2f}"

    return (
        LibraryBuilder()
        .add_polynomial(
            max_degree=3,
            include_constant=True,
            include_interactions=False,
        )
        .add_custom(powered_interaction, powered_interaction_name)
    )


def generate_data(
    *,
    a: float = 1.0,
    b: float = 3.0,
    alpha: float = 2.0,
    beta: float = 1.0,
    noise_level: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Generate two noisy trajectories and exact clean-system derivatives."""
    rng = np.random.default_rng(seed)

    system = Brussellator(a=a, b=b, alpha=alpha, beta=beta)
    time = np.arange(0.0, 25.0, 0.025)
    initial_conditions = [
        np.array([1.0, 1.0]),
        np.array([5, 3]), 
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


def build_training_data(
    time: np.ndarray,
    x_train: np.ndarray,
    x_dot: np.ndarray,
    initial_conditions: list[np.ndarray],
    library: LibraryBuilder,
) -> DataConstructor:
    """Create the DDMO data container for multiple trajectories."""
    return DataConstructor(
        x_train=x_train,
        time_array=time,
        x_dot=x_dot,
        library_object=library.build,
        x_0=initial_conditions,
        x_raw=x_train,
    )
  

def build_optimizer_settings() -> OptimizerSettings:
    """Configure the search over library, weak-form, and solver parameters."""
    return OptimizerSettings(
        library_bounds={
            "g1": (0., 3.0),
            "g2": (0., 3.0),
        },
        formulation_bounds={
            "degree": (2, 7),
            "stride": (5, 50),
            "window_length": (10, 100),
        },
        solver_bounds={
            "threshold": (0.5, 2),
            "alpha": (1, 10),
        },
        n_obj=2,
        n_gen=15,
        pop_size=100,
        algorithm="NSGA3",
        seed=42,
        verbose=True,
    )

def build_solver_settings() -> SolverSettings:
    """Configure the sparse-regression backend."""
    return SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "fit_intercept": False,
            "normalize_columns": False,
        },
        scale_features=True,
    )

def build_formulation_settings() -> FormulationSettings:
    """Configure the weak formulation used before optimization."""
    return FormulationSettings(
        formulation="weak",
        test_function="PolynomialWindow",
        degree=2,
        window_length=65,
        stride=50,
    )


def split_solution(
    solution: np.ndarray | None,
    n_trajectories: int,
    n_time: int,
) -> list[np.ndarray | None]:
    """Split the solver's stacked trajectory output into per-trajectory arrays."""
    if solution is None:
        return [None] * n_trajectories
    return [solution[i * n_time : (i + 1) * n_time] for i in range(n_trajectories)]


def make_plot(
    time: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray | None,
) -> None:
    """Plot both observed trajectories and the identified trajectories."""
    variable_names = ["x", "y"]
    predicted_blocks = split_solution(predicted, observed.shape[0], len(time))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for var_idx, name in enumerate(variable_names):
        ax = axes[var_idx]
        for traj_idx in range(observed.shape[0]):
            ax.plot(
                time,
                observed[traj_idx, :, var_idx],
                alpha=0.6,
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
    fig.suptitle("Custom-library Brusselator identification, two trajectories")
    fig.tight_layout()
    plt.show()


def main() -> None:
    """Run weak-form DDMO on two custom-library Brusselator trajectories."""
    time, x_train, x_dot, initial_conditions = generate_data(noise_level=0.1, 
                                                             alpha=2, beta=2, 
                                                             seed=42)
    library = build_custom_library()
    data = build_training_data(time, x_train, x_dot, initial_conditions, library)

    optimizer_settings = build_optimizer_settings()
    solver_settings = build_solver_settings()
    formulation_settings = build_formulation_settings()

    result = identify(
        data_instance=data,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings,
    )

    print("\nOptimized library/formulation/solver parameters:")
    print("Each row is one selected parameter vector from the Pareto front.")
    print(result.alphas)

    if len(result.solutions) > 0:
        make_plot(time, x_train, result.solutions[0])
    else:
        print("No solutions found.")
    
    print(result.coeff_matrices)


if __name__ == "__main__":
    main()
