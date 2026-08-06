# objective_functions.py
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import linregress
from sklearn import metrics


@dataclass(frozen=True)
class ObjectiveContext:
    """Inputs available to a user-defined optimization objective.

    Objective functions receive one context object rather than a changing list
    of positional arguments. This gives users access to the observed data and
    the complete solver result, including trajectories and coefficients.
    """

    data: Any
    solver_result: Any

    @property
    def solution(self):
        return self.solver_result.solution

    @property
    def coefficients(self):
        return self.solver_result.coefficients


ObjectiveFunction = Callable[[ObjectiveContext], float]


def trajectory_mse(context: ObjectiveContext) -> float:
    """Return the worst per-state trajectory mean squared error."""
    return compute_mse(context.data, context.solution)


def durbin_watson_deviation(context: ObjectiveContext) -> float:
    """Return the worst absolute residual Durbin-Watson deviation from two."""
    return compute_dw_coefficient(context.data, context.solution)


def active_coefficient_count(
    context: ObjectiveContext,
    *,
    tolerance: float = 1e-8,
) -> float:
    """Count fitted coefficients whose absolute value exceeds ``tolerance``."""
    coefficients = np.asarray(context.coefficients, dtype=float)
    return float(np.count_nonzero(np.abs(coefficients) > tolerance))


def make_active_coefficient_count(
    *,
    tolerance: float = 1e-8,
) -> ObjectiveFunction:
    """Create an active-coefficient objective with a custom zero tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    def objective(context: ObjectiveContext) -> float:
        return active_coefficient_count(context, tolerance=tolerance)

    objective.__name__ = "active_coefficient_count"
    return objective


DEFAULT_OBJECTIVES: dict[str, ObjectiveFunction] = {
    "trajectory_mse": trajectory_mse,
    "durbin_watson_deviation": durbin_watson_deviation,
}


def compute_dw_coefficient(data, sol) -> float:
    """Return the worst absolute Durbin-Watson deviation from two.

    A value of zero is ideal. The maximum is taken across all state variables
    and trajectories so that one autocorrelated residual series cannot be
    hidden by better-behaved series.
    """
    x_blocks = _trajectory_blocks(data.x_raw)
    sol_blocks = _solution_blocks(sol, x_blocks)
    deviations = np.zeros((data.dimension, len(x_blocks)))

    for i, (x_ref_block, x_pred_block) in enumerate(zip(x_blocks, sol_blocks)):
        for j in range(data.dimension):
            x_ref = x_ref_block[:, j]
            x_pred = x_pred_block[:, j]
            residuals = x_ref - x_pred
            if np.allclose(residuals, 0.0):
                deviation = 0.0
            else:
                dw_stat = float(sm.stats.stattools.durbin_watson(residuals))
                deviation = abs(dw_stat - 2.0)
            deviations[j, i] = deviation

    return float(np.max(deviations))


def compute_mse(data, sol):
    x_blocks = _trajectory_blocks(data.x_raw)
    sol_blocks = _solution_blocks(sol, x_blocks)
    score_fit_matrix = np.zeros((data.dimension, len(x_blocks)))

    for i, (x_ref_block, x_pred_block) in enumerate(zip(x_blocks, sol_blocks)):
        for j in range(data.dimension):
            x_ref = x_ref_block[:, j]
            x_pred = x_pred_block[:, j]
            score_fit_matrix[j, i] = metrics.mean_squared_error(x_ref, x_pred)

    score_fit = np.max(score_fit_matrix)

    return score_fit


def compute_ppcc_matrix(DataConstructor, sol, metric="quantiles"):
    x_blocks = _trajectory_blocks(DataConstructor.x_raw)
    sol_blocks = _solution_blocks(sol, x_blocks)
    ppcc_matrix = np.zeros((DataConstructor.dimension, len(x_blocks)))
    slope_qq_plot = np.zeros((DataConstructor.dimension, len(x_blocks)))
    intercept_qq_plot = np.zeros((DataConstructor.dimension, len(x_blocks)))
    # calculate the pij element of P
    for i, (x_ref_block, x_pred_block) in enumerate(zip(x_blocks, sol_blocks)):
        time_array = _time_vector_for_block(DataConstructor.time_array, i)
        for j in range(DataConstructor.dimension):
            x_ref = x_ref_block[:, j]
            x_pred = x_pred_block[:, j]

            residuals = x_ref - x_pred

            # De-trend the residuals, to compute the theoretical quantile
            slope_drift, intercept, _, _, _ = linregress(time_array, residuals)

            drift = slope_drift * time_array + intercept
            corrected_residuals = residuals - drift

            # Compute the PP plot btw the theoretical quantile, and the residuals
            if metric == "quantiles":
                pp_plot = stats.probplot(
                    residuals,
                    dist="norm",
                    sparams=(corrected_residuals.mean(), corrected_residuals.std()),
                    fit=False,
                )
                x_axis = pp_plot[0]
                y_axis = pp_plot[1]
            elif metric == "ppf":
                quantiles = stats.norm.ppf(
                    stats.rankdata(corrected_residuals) / (len(residuals) + 1)
                )
                x_axis = quantiles
                y_axis = residuals

            slope, intercept = np.polyfit(x_axis, y_axis, 1)
            ppcc_matrix[j, i] = stats.pearsonr(x_axis, y_axis)[0]
            slope_qq_plot[j, i] = slope
            intercept_qq_plot[j, i] = intercept
        ppcc_stats = np.min(ppcc_matrix)
        intercept_qq_out = np.max(abs(intercept_qq_plot))

    return abs(1 - ppcc_stats), slope_qq_plot, intercept_qq_out


def _trajectory_blocks(values):
    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            return [values]
        if values.ndim == 3:
            return [values[i] for i in range(values.shape[0])]

    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return [np.asarray(block) for block in values]

    return [np.atleast_2d(np.asarray(values))]


def _solution_blocks(sol, reference_blocks):
    sol_array = np.asarray(sol)
    if sol_array.ndim == 3:
        return [sol_array[i] for i in range(sol_array.shape[0])]

    lengths = [block.shape[0] for block in reference_blocks]
    split_points = np.cumsum(lengths)[:-1]
    return list(np.split(sol_array, split_points, axis=0))


def _time_vector_for_block(time_array, block_index):
    time_values = np.asarray(time_array)
    if time_values.ndim == 1:
        return time_values
    return time_values[block_index]
