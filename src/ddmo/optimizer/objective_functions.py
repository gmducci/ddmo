# objective_functions.py
from collections.abc import Sequence

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import linregress
from sklearn import metrics


def compute_dw_coefficient(DataConstructor, sol):
    """
    Computes the Durbin-Watson statistic for residuals between reference and predicted values.
    Returns the maximum absolute deviation from 2 across all dimensions and initial conditions.
    """
    x_blocks = _trajectory_blocks(DataConstructor.x_raw)
    sol_blocks = _solution_blocks(sol, x_blocks)
    dw_matrix = np.zeros(
        (DataConstructor.dimension, len(x_blocks))
    )

    for i, (x_ref_block, x_pred_block) in enumerate(zip(x_blocks, sol_blocks)):
        for j in range(DataConstructor.dimension):
            x_ref = x_ref_block[:, j]
            x_pred = x_pred_block[:, j]
            residuals = x_ref - x_pred
            dw_matrix[j, i] = np.abs(sm.stats.stattools.durbin_watson(residuals))

    dw_stat = np.min(dw_matrix)

    dw_measure = 2 - dw_stat
    return dw_measure


def compute_mse(DataConstructor, sol):
    x_blocks = _trajectory_blocks(DataConstructor.x_raw)
    sol_blocks = _solution_blocks(sol, x_blocks)
    score_fit_matrix = np.zeros(
        (DataConstructor.dimension, len(x_blocks))
    )

    for i, (x_ref_block, x_pred_block) in enumerate(zip(x_blocks, sol_blocks)):
        for j in range(DataConstructor.dimension):
            x_ref = x_ref_block[:, j]
            x_pred = x_pred_block[:, j]
            score_fit_matrix[j, i] = metrics.mean_squared_error(x_ref, x_pred)

    score_fit = np.max(score_fit_matrix)

    return score_fit


def compute_ppcc_matrix(DataConstructor, sol, metric="quantiles"):
    x_blocks = _trajectory_blocks(DataConstructor.x_raw)
    sol_blocks = _solution_blocks(sol, x_blocks)
    ppcc_matrix = np.zeros(
        (DataConstructor.dimension, len(x_blocks))
    )
    slope_qq_plot = np.zeros(
        (DataConstructor.dimension, len(x_blocks))
    )
    intercept_qq_plot = np.zeros(
        (DataConstructor.dimension, len(x_blocks))
    )
    # calculate the pij element of P
    for i, (x_ref_block, x_pred_block) in enumerate(zip(x_blocks, sol_blocks)):
        time_array = _time_vector_for_block(DataConstructor.time_array, i)
        for j in range(DataConstructor.dimension):
            x_ref = x_ref_block[:, j]
            x_pred = x_pred_block[:, j]

            residuals = x_ref - x_pred

            # De-trend the residuals, to compute the theoretical quantile
            slope_drift, intercept, _, _, _ = linregress(
                time_array, residuals
            )

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
