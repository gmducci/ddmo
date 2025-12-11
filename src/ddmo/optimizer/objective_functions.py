# objective_functions.py
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
    dw_matrix = np.zeros(
        (DataConstructor.dimension, DataConstructor.number_initial_conditions)
    )
    x_raw = np.vstack(DataConstructor.x_raw)
    t_len = len(x_raw)  # Assuming all initial conditions have the same length

    for i in range(DataConstructor.number_initial_conditions):
        for j in range(DataConstructor.dimension):
            x_ref = x_raw[i * t_len : (i + 1) * t_len, j]
            x_pred = sol[i * t_len : (i + 1) * t_len, j]
            residuals = x_ref - x_pred
            dw_matrix[j, i] = np.abs(sm.stats.stattools.durbin_watson(residuals))

    dw_stat = np.min(dw_matrix)

    dw_measure = 2 - dw_stat
    return dw_measure


def compute_mse(DataConstructor, sol):
    score_fit_matrix = np.zeros(
        (DataConstructor.dimension, DataConstructor.number_initial_conditions)
    )
    for i in range(DataConstructor.number_initial_conditions):
        for j in range(DataConstructor.dimension):
            x_raw = np.vstack(DataConstructor.x_raw)
            x_ref = x_raw[i * len(x_raw) : (i + 1) * len(x_raw), j]
            x_pred = sol[i * len(x_raw) : (i + 1) * len(x_raw), j]
            score_fit_matrix[j, i] = metrics.mean_squared_error(x_ref, x_pred)

    score_fit = np.max(score_fit_matrix)

    return score_fit


def compute_ppcc_matrix(DataConstructor, sol, metric="quantiles"):
    ppcc_matrix = np.zeros(
        (DataConstructor.dimension, DataConstructor.number_initial_conditions)
    )
    slope_qq_plot = np.zeros(
        (DataConstructor.dimension, DataConstructor.number_initial_conditions)
    )
    intercept_qq_plot = np.zeros(
        (DataConstructor.dimension, DataConstructor.number_initial_conditions)
    )
    # calculate the pij element of P
    for i in range(DataConstructor.number_initial_conditions):
        for j in range(DataConstructor.dimension):
            x_ref = DataConstructor.x_raw[
                i * len(DataConstructor.time_array) : (i + 1)
                * len(DataConstructor.time_array),
                j,
            ]
            x_pred = sol[
                i * len(DataConstructor.time_array) : (i + 1)
                * len(DataConstructor.time_array),
                j,
            ]

            residuals = x_ref - x_pred

            # De-trend the residuals, to compute the theoretical quantile
            slope_drift, intercept, _, _, _ = linregress(
                DataConstructor.time_array, residuals
            )

            drift = slope_drift * DataConstructor.time_array + intercept
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
            ppcc_matrix[i, j] = stats.pearsonr(x_axis, y_axis)[0]
            slope_qq_plot[i, j] = slope
            intercept_qq_plot[i, j] = intercept
        ppcc_stats = np.min(ppcc_matrix)
        intercept_qq_out = np.max(abs(intercept_qq_plot))

    return abs(1 - ppcc_stats), slope_qq_plot, intercept_qq_out
