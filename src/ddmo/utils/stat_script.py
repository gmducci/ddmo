import random
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

path = '/home/ducci/PROJECT/ddmo/results/paper_batch6/'
num_cases = 85#100  # Assuming you have cases from 0 to 99

solutions_list = []
coefficients = []
alpha = []
for i in range(num_cases):
    case_folder = f'case_{i}/'
    try:
        sol = np.load(path + case_folder + 'solutions_nsga.npy')
        hill_coeff =  np.load(path + case_folder + 'alphas.npy')
        coeff_matrix = np.load(path + case_folder + 'coeff_matrices.npy')

        n_pareto = len(sol)
        print(n_pareto)
        random_solution = random.randint(0, n_pareto -1)
        sol = sol[random_solution]
        hill_coeff = hill_coeff[random_solution]
        coeff_matrix = coeff_matrix[random_solution]
        coeff_matrix[abs(coeff_matrix) < 0.001] = 0
        alpha.append(hill_coeff)
        solutions_list.append(sol)
        coefficients.append(coeff_matrix)
        print(f"Loaded solutions from {case_folder}")
    except FileNotFoundError:
        print(f"No solutions found in {case_folder}")

solution_array = np.asarray(solutions_list)
alpha = np.asarray(alpha)
coefficients = np.asarray(coefficients)
coefficients[abs(coefficients) < 0.0001] = 0
# plt.figure(figsize=(8, 6))
# parts = plt.violinplot(alpha[:, 0], vert=True, showmeans=False, showmedians=False, showextrema=False)
#
# # Customizing the violin plot stroke line
# for pc in parts['bodies']:
#     pc.set_edgecolor('black')
#     pc.set_linewidth(1)
# plt.scatter(np.ones_like(alpha[:, 0]), alpha[:, 0],  color='red', alpha=0.5)
# # Adding labels and title
# plt.xlabel('Alpha Values')
# plt.title('Violin Plot of Alpha')
# # Display the plot
# plt.show()

true_coeffs = np.zeros_like(coefficients)
true_coeffs = true_coeffs.T
true_coeffs[0, 0] = -0.2

true_coeffs[1, 0] = 1

true_coeffs[1, 1] = -0.15
true_coeffs[2, 1] = 1
true_coeffs[2, 2] = -0.1
true_coeffs[0, 3] = 1

fig, axes = plt.subplots(nrows=3, ncols=11, figsize=(16, 9))

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
custom_labels = ['$x$', '$y$', '$z$', r'$\frac{1}{1 + z^{\gamma_1}}$', r'$\exp(-\gamma_2 z)$', '$x^2$', '$y^2$', '$z^2$', '', '', '']
custom_y = [r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']

# Add titles to subplots
for i in range(3):
    for j in range(11):
        boxplot = axes[i, j].boxplot(coefficients[:, j, i], showfliers=True)
        axes[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # Remove x labels except for the last row
        if j == 0:
            axes[i, j].set_ylabel(custom_y[i], fontsize = 25, labelpad = 20, rotation = 0)

        if i < 2:
            axes[i, j].set_xticklabels([])

        # Customize x-labels for the last row
        else:
            # Assuming you have defined custom x-labels in 'custom_labels'
            # Ensure the number of custom labels matches the number of ticks
            #axes[i, j].set_xticks(range(1, len(custom_labels) + 1))
            axes[i, j].set_xticks([])
            axes[i, j].set_xlabel(custom_labels[j], fontsize = 25, labelpad = 10)

        axes[i,j].hlines(true_coeffs[i, j],  0.75,  1.25, colors='r', linestyles='dashed', lw = 2)
        for median in boxplot['medians']:
            median.set_color('black')  # Set color to black
            median.set_linewidth(2)  # Set thickness to desired value
        # Customizing the violin plot stroke line
        # for pc in parts['bodies']:
        #     pc.set_edgecolor('black')
        #     pc.set_linewidth(1)
# Show the plot
plt.subplots_adjust(top=0.9,
bottom=0.1,
left=0.06,
right=0.95,
hspace=0.4,
wspace=0.6)
plt.savefig('stat_robustness.pdf', format = 'pdf')
plt.show()
