
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import linregress

warnings.filterwarnings("ignore")


matplotlib.use('Qt5Agg')
params = {
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'lines.linewidth': 3,
    'figure.figsize': [8, 6],
    'axes.grid': False,  # Turn off grid    'xtick.major.width': 3,
    'ytick.major.width': 3,
    'xtick.major.width': 3,
    'ytick.minor.width': 1.5,
    'xtick.minor.width': 1.5,

    'axes.linewidth': 2,  # Spine line width
    'axes.spines.top': True,  # Hide top spine
    'axes.spines.right': True,  # Hide right spine
}

params['xtick.major.size'] = 5
params['xtick.minor.size'] = 5
params['ytick.minor.size'] = 5

params['ytick.major.size'] = 5
plt.rcParams.update(params)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Calibri",
})
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'serif'

results_path = '/home/ducci/PROJECT/MyWrittenDocs/paper1/Draft_paper/figures/PPCC_DW_analysis'

time_array = np.load(results_path + '/time_array.npy')
res_F = np.load(results_path + '/res.npy')
x_train = np.load(results_path + '/x_train_clean.npy')
x_train_noise = np.load(results_path + '/x_gt.npy')
x_dot_train = np.load(results_path + '/x_dot_gt.npy')
#x_wt = np.load(results_path + '/wavelet_derivatives.npy')
#x_dot_hat = np.load(results_path + '/x_dot_kalman.npy')
#x_hat = np.load(results_path + '/x_kalman.npy')
solutions = np.load(results_path + '/solutions_nsga.npy')
coeff_matrices = np.load(results_path + '/coeff_matrices.npy')
alphas = np.load(results_path + '/alphas.npy')
print(alphas)
dw_stats = np.load(results_path + '/dw_stats.npy')

index = np.argmax(dw_stats)

x_ref = x_train_noise[0:500, 1]
x_pred =  solutions[index][0:500, 1]
residuals =x_ref - x_pred
quantiles = stats.norm.ppf((np.argsort(np.argsort(residuals)) + 1) / (len(residuals) + 1))
correlation_coefficient = np.corrcoef(residuals, quantiles)[0, 1]

#########################################
# Figure 1
#########################################

fig = plt.figure(figsize = (15, 10))
ax = plt.subplot(2, 3, 1)
ax.plot(time_array, x_ref)
ax.plot(time_array, x_pred)
ax.set_xlabel('Time')
ax.set_ylabel('x, $x_{GT}$')

ax1 = plt.subplot(2, 3, 2)
ax1.plot(time_array, residuals)
ax1.set_xlabel('Time')
ax1.set_ylabel('residuals')
ax2 = plt.subplot(2, 3, 3)
plot_qq = sm.qqplot(residuals, fit = True, line = '45', ax=ax2)
ax2.set_aspect('equal')
dw = sm.stats.stattools.durbin_watson(residuals)
ax2.set_title(f'PPCC = {correlation_coefficient:.4f}\nDW stats = {dw:.4f}')

index_1 = np.argmin(dw_stats)

x_ref_1 = x_train_noise[0:500, 1]
x_pred_1 =  solutions[index_1][0:500, 1]
residuals_1 =x_ref_1 - x_pred_1
quantiles_1 = stats.norm.ppf((np.argsort(np.argsort(residuals_1)) + 1) / (len(residuals_1) + 1))
correlation_coefficient_1 = np.corrcoef(residuals_1, quantiles_1)[0, 1]

ax3 = plt.subplot(2, 3, 4)
ax3.plot(time_array, x_ref_1)
ax3.plot(time_array, x_pred_1)
ax3.set_xlabel('Time')
ax3.set_ylabel('x, $x_{GT}$')

ax4 = plt.subplot(2, 3, 5)
ax4.plot(time_array, residuals_1)
ax4.set_xlabel('Time')
ax4.set_ylabel('residuals')
ax5 = plt.subplot(2, 3, 6)
plot_qq = sm.qqplot(residuals_1, fit = True, line = '45', ax=ax5)
ax5.set_aspect('equal')
dw = sm.stats.stattools.durbin_watson(residuals_1)
ax5.set_title(f'PPCC = {correlation_coefficient_1:.4f}\nDW stats = {dw:.4f}')

plt.subplots_adjust(top=0.88,
bottom=0.11,
left=0.05,
right=0.995,
hspace=0.37,
wspace=0.2)
plt.show()

#########################################
# Figure 2 Detrend
#########################################

slope, intercept, _, _, _ = linregress(time_array, residuals)

drift = slope * time_array + intercept
corrected_residuals = residuals - drift

# Compute drift parameters for residuals_1
slope_1, intercept_1, _, _, _ = linregress(time_array, residuals_1)
drift_1 = slope_1 * time_array + intercept_1
corrected_residuals_1 = residuals_1 - drift_1

# Plotting the original and corrected residuals for both distributions

#########################################
# Figure 3 Gaussian
#########################################

quantiles = stats.norm.ppf(stats.rankdata(corrected_residuals) / (len(residuals) + 1))
ppcc = stats.pearsonr(quantiles, residuals)[0]
quantiles_1 = stats.norm.ppf(stats.rankdata(corrected_residuals_1) / (len(residuals_1) + 1))
ppcc_1 = stats.pearsonr(quantiles_1, residuals_1)[0]

fig = plt.figure(figsize = (15, 10))
ax = plt.subplot(2, 3, 1)
ax.plot(time_array, x_ref)
ax.plot(time_array, x_pred)
ax.set_xlabel('Time')
ax.set_ylabel('x, $x_{GT}$')

ax1 = plt.subplot(2, 3, 2)
ax1.plot(time_array, residuals)
ax1.set_xlabel('Time')
ax1.set_ylabel('residuals')
dw = sm.stats.stattools.durbin_watson(residuals)
ax1.set_title(f'DW stats = {dw:.4f}')

ax2 = plt.subplot(2, 3, 3)
ax2.scatter(quantiles, residuals)
ax2.set_title(f'PPCC = {ppcc:.4f}')
ax2.set_ylabel('residuals')
ax2.set_xlabel('Theoretical percentile')


index_1 = np.argmin(dw_stats)

ax3 = plt.subplot(2, 3, 4)
ax3.plot(time_array, x_ref_1)
ax3.plot(time_array, x_pred_1)
ax3.set_xlabel('Time')
ax3.set_ylabel('x, $x_{GT}$')


ax4 = plt.subplot(2, 3, 5)
ax4.plot(time_array, residuals_1)
ax4.set_xlabel('Time')
ax4.set_ylabel('residuals')
dw_1 = sm.stats.stattools.durbin_watson(residuals_1)
ax4.set_title(f'DW stats = {dw_1:.4f}')

ax5 = plt.subplot(2, 3, 6)
ax5.scatter(quantiles_1, residuals_1)
ax5.set_ylabel('residuals')
ax5.set_xlabel('Theoretical percentile')
ax5.set_title(f'PPCC = {ppcc_1:.4f}')

plt.subplots_adjust(top=0.88,
bottom=0.11,
left=0.05,
right=0.995,
hspace=0.37,
wspace=0.3)
# plt.savefig('/home/ducci/PROJECT/MyPresentations/DynamicsDays24/newfigures/residuals.png')
plt.show()
