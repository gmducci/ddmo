"""Goodwin oscillator example using the weak formulation.

This script demonstrates a complete DDMO workflow on the benchmark Goodwin
oscillator data:

1. load benchmark trajectories;
2. construct a candidate library;
3. configure weak-form model identification;
4. optimize library and weak-form parameters;
5. compare the identified trajectory with the observed data.

Run from the repository root with:

    PYTHONPATH=src python examples/ex_goodwin_weak.py

or after installing the package in editable mode:

    python examples/ex_goodwin_weak.py
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


def load_goodwin_benchmark():
    """Load and downsample the benchmark Goodwin oscillator data."""
    x_0_train = np.load(BENCHMARK_FOLDER / "x_0.npy")
    x_noisy = np.load(BENCHMARK_FOLDER / "x_train_noise.npy")
    time_array = np.load(BENCHMARK_FOLDER / "time.npy")

    return {
        "x_0": [x_0_train],
        "x_train": x_noisy,
        "x_raw": x_noisy,
        "time_array": time_array,
    }


def build_candidate_library():
    """Define the candidate terms available to the discovered model."""
    return (
        LibraryBuilder()
        .add_polynomial(max_degree=2, include_interactions=True)
        .add_hill(exponent_param="gamma_1")
        .add_exponential(rate_param="gamma_2")
    )


def build_training_data(data: dict, library: LibraryBuilder) -> DataConstructor:
    """Create the DDMO data container used by the identification pipeline."""
    return DataConstructor(
        x_train=data["x_train"],
        time_array=data["time_array"],
        library_object=library.build,
        x_0=data["x_0"],
        x_raw=data["x_raw"],
    )


def build_optimizer_settings() -> OptimizerSettings:
    """Configure the multi-objective search over model-discovery parameters."""
    library_bounds = {
        "gamma_1": (8.4, 9.6),
        "gamma_2": (1.0, 10.0),
    }

    formulation_bounds = {
        "degree": (2, 7),
        "stride": (50, 250),
        "window_length": (127, 801),
    }

    return OptimizerSettings(
        library_bounds=library_bounds,
        formulation_bounds=formulation_bounds,
        n_obj=2,
        n_gen=10,
        algorithm="NSGA3",
        pop_size=50,
        seed=RANDOM_SEED,
    )


def build_solver_settings() -> SolverSettings:
    """Configure the sparse-regression backend."""
    return SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "normalize_columns": False,
            "alpha": 0.05,
            "threshold": 0.1,
        },
        scale_features=True,
    )


def build_formulation_settings() -> FormulationSettings:
    """Configure the initial weak formulation used before optimization."""
    return FormulationSettings(
        formulation="weak",
        test_function="PolynomialWindow",
        degree=2,
        window_length=65,
        stride=50,
    )


def plot_solution(result, observed: np.ndarray) -> None:
    """Plot the first identified trajectory against the observed data."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(result.solutions[0], label="identified trajectory")
    ax.plot(observed, alpha=0.5, linestyle="--", label="observed noisy data")

    ax.set_title("Goodwin oscillator: weak-form DDMO identification")
    ax.set_xlabel("time index")
    ax.set_ylabel("state value")
    ax.legend()
    fig.tight_layout()
    plt.show()


def run() -> None:
    """Run the Goodwin weak-form tutorial.

    This function contains the same code previously guarded by
    ``if __name__ == '__main__'`` so the example is easier to read and
    can be imported and executed programmatically (e.g. from a notebook).
    """
    data = load_goodwin_benchmark()
    library = build_candidate_library()
    training_data = build_training_data(data, library)

    optimizer_settings = build_optimizer_settings()
    solver_settings = build_solver_settings()
    formulation_settings = build_formulation_settings()

    result = identify(
        data_instance=training_data,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings,
    )

    print("Selected parameter vectors / alphas:")
    print(result.alphas)

    plot_solution(result, observed=data["x_raw"])
