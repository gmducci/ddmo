"""A small, reusable DDMO tutorial using the Goodwin oscillator.

Run from the repository root after installing the package:

    python examples/quickstart/goodwin_template.py

The oscillator supplies example observations. For another problem, replace the
data-generation block with your measured trajectories and initial conditions.
The Library, Formulation, and Solver blocks are independent snippets that can
also be used in a methods description.
"""

import numpy as np

from ddmo import (
    DataConstructor,
    FormulationSettings,
    ODEDataGenerator,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.equations.goodwin_eqs import GoodwinModel
from ddmo.library_functions import LibraryBuilder


def main() -> None:
    # Example data: two trajectories for fitting and one for choosing a model.
    # The clean simulated data are used only to create noisy observations.
    time = np.arange(0.0, 100.1, 0.2)
    initial_conditions = [
        np.array([0.5, 1.0, 1.0]),
        np.array([0.2, 1.4, 0.8]),
        np.array([0.8, 0.6, 1.5]),
    ]
    observations = ODEDataGenerator(
        GoodwinModel(n=9.5).dynamics,
        initial_conditions=initial_conditions,
        time_array=time,
        noise_level=0.05,
        seed=7,
        clip_min=1e-8,  # Fractional Hill powers require positive states.
    ).generate().x_noisy

    data = DataConstructor(
        x_train=observations[:2],
        x_raw=observations[:2],
        time_array=time,
        x_0=initial_conditions[:2],
    )

    # 1. Library: candidate terms for the right-hand side of each equation.
    # DDMO will optimize gamma_1 within the bounds given below.
    library = (
        LibraryBuilder()
        .add_polynomial(max_degree=1, include_constant=False)
        .add_hill(exponent_param="gamma_1")
        .add_exponential(exponent_param="gamma_2")
    )

    # 2. Formulation: weak form avoids estimating pointwise derivatives.
    formulation = FormulationSettings(
        formulation="weak",
        test_function="PolynomialWindow",
        degree=2,
        window_length=61,
        stride=15,
    )

    # 3. Solver: sparse regression gives coefficients for the candidate terms.
    solver = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "fit_intercept": False,
            "normalize_columns": False,
        },
        scale_features=True,
    )

    # Search ranges are separate from the three definitions above.
    # The small population and generation count keep this tutorial quick.
    optimizer = OptimizerSettings(
        library_bounds={"gamma_1": (6.0, 12.0)},
        formulation_bounds={
            "degree": (2, 4),
            "window_length": (41, 101),
            "stride": (10, 30),
        },
        solver_bounds={
            "threshold": (0.001, 0.2),
            "alpha": (1e-6, 0.1),
        },
        algorithm="NSGA2",
        pop_size=12,
        n_gen=3,
        seed=7,
        verbose=False,
    )

    result = identify(
        data_instance=data,
        library_object=library.build,
        formulation_settings=formulation,
        solver_settings=solver,
        optimizer_settings=optimizer,
    )

    # The optimizer can return several Pareto models. Choose one using the
    # held-out noisy trajectory, without using the clean simulated trajectory.
    best_index = None
    best_error = float("inf")
    for index, model in enumerate(result.models):
        if model is None:
            continue
        try:
            predicted = model.simulate(initial_conditions[2], time)
            error = float(np.sqrt(np.mean((predicted - observations[2]) ** 2)))
        except Exception:
            continue  # Some fitted ODEs cannot be integrated over the full time span.
        if np.isfinite(error) and error < best_error:
            best_index, best_error = index, error

    if best_index is None:
        raise RuntimeError("No fitted model completed the validation trajectory.")

    print(f"Selected model: {best_index + 1} of {len(result.models)}")
    print(f"Validation RMSE against noisy observations: {best_error:.4f}")
    print("Selected parameters:")
    for name, value in zip(optimizer.all_bounds, result.alphas[best_index]):
        print(f"  {name}: {float(value):.4g}")
    print("Identified equations:")
    for equation in result.symbolic_equations(
        index=best_index,
        input_features=["x", "y", "z"],
        state_names=["x", "y", "z"],
        coefficient_threshold=1e-6,
    ):
        print(f"  {equation}")


if __name__ == "__main__":
    main()
