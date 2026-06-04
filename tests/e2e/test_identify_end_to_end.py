import numpy as np

from ddmo import (
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.library_functions import LibraryBuilder


def test_identify_strong_formulation_end_to_end():
    """Run the public workflow from data construction to identified solution."""
    time = np.linspace(0.0, 2.0, 40)
    x_train = np.exp(-0.5 * time).reshape(-1, 1)
    x_dot = -0.5 * x_train

    library = LibraryBuilder().add_polynomial(
        max_degree=1,
        include_constant=False,
    )

    data = DataConstructor(
        x_train=x_train,
        time_array=time,
        x_dot=x_dot,
        x_0=[x_train[0]],
        x_raw=x_train,
    )

    optimizer_settings = OptimizerSettings(
        library_bounds={},
        formulation_bounds={},
        solver_bounds={"threshold": (0.001, 0.01)},
        n_obj=2,
        n_gen=1,
        pop_size=4,
        algorithm="NSGA2",
        seed=1,
        verbose=False,
    )

    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "alpha": 0.0,
            "fit_intercept": False,
            "normalize_columns": False,
        },
        scale_features=True,
    )

    formulation_settings = FormulationSettings(formulation="strong")

    result = identify(
        data_instance=data,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings,
        library_object=library.build,
    )

    assert result.objective_space.shape == (optimizer_settings.pop_size, 2)
    assert np.all(np.isfinite(result.objective_space))

    assert len(result.alphas) == optimizer_settings.pop_size
    assert all(len(alpha) == optimizer_settings.n_var for alpha in result.alphas)

    assert len(result.solutions) == optimizer_settings.pop_size
    assert all(solution.shape == x_train.shape for solution in result.solutions)
    assert all(np.all(np.isfinite(solution)) for solution in result.solutions)

    assert len(result.coeff_matrices) == optimizer_settings.pop_size
    assert all(coefficients.shape == (1, 1) for coefficients in result.coeff_matrices)
    assert any(
        np.isclose(coefficients[0, 0], -0.5, atol=1e-8)
        for coefficients in result.coeff_matrices
    )

    assert len(result.theta_matrices) == optimizer_settings.pop_size
    assert all(theta.shape == x_train.shape for theta in result.theta_matrices)
