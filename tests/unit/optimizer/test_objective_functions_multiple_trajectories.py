import numpy as np

from ddmo.optimizer.objective_functions import (
    DEFAULT_OBJECTIVES,
    ObjectiveContext,
    active_coefficient_count,
    compute_dw_coefficient,
    compute_mse,
    make_active_coefficient_count,
)
from ddmo.solver.result import SolverResult
from ddmo.utils.data_constructor import DataConstructor


def test_compute_mse_slices_stacked_solution_by_trajectory_length():
    x_raw = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[10.0], [20.0], [30.0]],
        ]
    )
    sol = np.vstack(
        [
            x_raw[0],
            x_raw[1] + 2.0,
        ]
    )
    data = DataConstructor(
        x_train=x_raw,
        time_array=np.arange(3, dtype=float),
        x_0=[x_raw[0, 0], x_raw[1, 0]],
        x_raw=x_raw,
    )

    assert compute_mse(data, sol) == 4.0


def test_durbin_watson_returns_worst_absolute_deviation_from_two():
    residuals = np.array(
        [
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 1.0],
            [0.0, -1.0],
        ]
    )
    data = DataConstructor(
        x_train=residuals,
        time_array=np.arange(4, dtype=float),
        x_0=[residuals[0]],
        x_raw=residuals,
    )

    # The two residual series have DW deviations 0.5 and 1.0 respectively.
    assert compute_dw_coefficient(data, np.zeros_like(residuals)) == 1.0


def test_durbin_watson_treats_a_perfect_fit_as_zero_deviation():
    values = np.arange(4, dtype=float).reshape(-1, 1)
    data = DataConstructor(
        x_train=values,
        time_array=np.arange(4, dtype=float),
        x_0=[values[0]],
        x_raw=values,
    )

    assert compute_dw_coefficient(data, values.copy()) == 0.0


def test_active_coefficient_count_is_optional_and_tolerance_aware():
    values = np.ones((3, 1))
    data = DataConstructor(
        x_train=values,
        time_array=np.arange(3, dtype=float),
        x_0=[values[0]],
        x_raw=values,
    )
    solver_result = SolverResult(
        solution=values,
        coefficients=np.array([[1.0, 1e-9], [0.0, -0.25]]),
        opt_vars=(),
        model=None,
    )
    context = ObjectiveContext(data=data, solver_result=solver_result)

    assert "active_coefficients" not in DEFAULT_OBJECTIVES
    assert active_coefficient_count(context) == 2.0
    assert make_active_coefficient_count(tolerance=0.5)(context) == 1.0
