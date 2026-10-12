"""Brusselator example with custom library functions.

This example demonstrates how to extend the DDMO candidate library with
user-defined functions. The target system is a two-dimensional Brusselator-like
ODE containing the nonlinear interaction term x^alpha * y^beta.

The example combines:

1. built-in identity terms, x and y;
2. a built-in constant term from a polynomial library;
3. a custom parameterized interaction term, x^g1 * y^g2.

The exponents g1 and g2 are treated as library parameters and optimized by
DDMO. For the default synthetic system, the expected values are close to:

    g1 ~= 2
    g2 ~= 1

Run from the repository root with:

    PYTHONPATH=src python examples/tutorial_custom_library_brusselator.py

or after installing the package in editable mode:

    python examples/tutorial_custom_library_brusselator.py

This compact tutorial trains on one synthetic trajectory. For a real
identification study, use several trajectories or validation initial
conditions to check that the discovered model generalizes beyond the training
path.
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
    """Create a library with built-in and user-defined functions.

    The custom function must accept the state variables first and the optimizer
    parameter dictionary last. Here, DDMO/PySINDy will pass x and y as the two
    state variables, while the optimizer supplies g1 and g2.
    """

    def powered_interaction(x: np.ndarray, y: np.ndarray, params: dict[str, float]):
        """Parameterized interaction term x^g1 * y^g2."""
        g1 = params["g1"]
        g2 = params["g2"]
        return (x**g1) * (y**g2)

    def powered_interaction_name(x: str, y: str, params: dict[str, float]) -> str:
        """Readable name for the custom interaction term."""
        g1 = params["g1"]
        g2 = params["g2"]
        return f"{x}^{g1:.2f}*{y}^{g2:.2f}"

    return (
        LibraryBuilder()
        # Constant + linear terms. We disable interactions here because the
        # nonlinear interaction is supplied explicitly by the custom function.
        .add_polynomial(
            max_degree=1,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Generate noisy observations and clean-system derivatives.

    The weak formulation does not require users to estimate derivatives from
    noisy data. The exact derivative array is included here because synthetic
    examples can provide it and some solver/objective paths may consume it.
    """
    rng = np.random.default_rng(seed)

    system = Brussellator(a=a, b=b, alpha=alpha, beta=beta)
    time = np.arange(0.0, 25.0, 0.05)
    x0 = [1.0, 1.0]

    x_clean = odeint(system.dynamics, x0, time)
    x_dot = np.asarray([system.dynamics(state, t) for state, t in zip(x_clean, time)])

    if noise_level > 0:
        scale = np.std(x_clean, axis=0)
        noise = rng.normal(loc=0.0, scale=noise_level, size=x_clean.shape) * scale
        x_observed = x_clean + noise
    else:
        x_observed = x_clean.copy()

    return time, x_observed, x_dot, x0


def make_plot(
    time: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray | None,
) -> None:
    """Plot observed and identified trajectories."""
    variable_names = ["x", "y"]

    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(variable_names):
        plt.subplot(2, 1, idx + 1)
        plt.plot(time, observed[:, idx], label=f"observed {name}", alpha=0.7)

        if predicted is not None:
            n = min(len(time), len(predicted))
            plt.plot(
                time[:n],
                predicted[:n, idx],
                linestyle="--",
                label=f"identified {name}",
            )

        plt.ylabel(name)
        plt.legend()

    plt.xlabel("time")
    plt.suptitle("Custom-library Brusselator identification")
    plt.tight_layout()
    plt.show()


time, x_train, x_dot, x0 = generate_data(noise_level=0.1, seed=42)
library = build_custom_library()

data = DataConstructor(
    x_train=x_train,
    time_array=time,
    x_dot=x_dot,
    x_0=[x0],
    x_raw=x_train,
)

# These names must match the keys read inside the custom function:
# params["g1"] and params["g2"].
optimizer_settings = OptimizerSettings(
    library_bounds={
        "g1": (1.0, 3.0),
        "g2": (0.5, 2.0),
    },
    formulation_bounds={
        "degree": (2, 7),
        "stride": (10, 100),
        "window_length": (30, 240),
    },
    solver_bounds={
        "threshold": (0.001, 0.2),
        "alpha": (0.0001, 0.1),
    },
    n_obj=2,
    n_gen=15,
    pop_size=40,
    algorithm="NSGA3",
    seed=42,
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
    window_length=65,
    stride=50,
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
