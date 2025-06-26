import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def plot_coefficients(coeff_matrix, lib_names, fig):

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Sample DataFrame with random data

    df = pd.DataFrame(np.round(coeff_matrix.T, 2))
    df.index = ['$\dot{x}$', '$\dot{y}$', '$\dot{z}$' ]#lib_names
    df.columns = lib_names#['$\dot{x}$', '$\dot{y}$', '$\dot{z}$' ]

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Create a heatmap plot
    #plt.figure(figsize=(8, 6))

    rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
    export_fig = sns.heatmap(df, cmap=rdgn, center=0.00, annot=True,
                linewidths=1.3, linecolor='black', cbar=False, xticklabels=False, yticklabels=False)
    #plt.show()

    return export_fig

def plot_grid(time_array, x_train_noise, solutions, res,
              coeff_matrices, lib_names, index, alpha, noise_level):
    fig = plt.figure(figsize=(12, 6))

    # Define grid for subplot placement
    grid = (4, 4)
    # Top Plot (Full Width)
    ax1 = plt.subplot2grid(grid, (0, 0), rowspan=2, colspan=4)
    ax1.plot(time_array, x_train_noise[:, 0], label='x', color='r', alpha=.5)
    ax1.plot(time_array, x_train_noise[:, 1], label='y', color='g', alpha=.5)
#    ax1.plot(time_array, x_train_noise[:, 2], label='z', color='b', alpha=.5)
    ax1.plot(time_array, solutions[:, 0], color='k', lw=2.)
    ax1.plot(time_array, solutions[:, 1], color='k', lw=2.)
#    ax1.plot(time_array, solutions[:, 2], color='k', lw=2.)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("x, y, z")
    plt.legend()

    # Bottom Left Plot
    ax2 = plt.subplot2grid(grid, (2, 0), rowspan=2, colspan=2)
    ax2.scatter(res.F[:, 0], res.F[:, 1], color='g')
    ax2.scatter(res.F[index, 0], res.F[index, 1], color='red', marker=(5, 2))
    ax2.set_ylabel('1 - PPCC')
    ax2.set_xlabel('Fit score')
    plt.title('Pareto front')

    # Bottom Right Plot
    ax3 = plt.subplot2grid(grid, (2, 2), rowspan=2, colspan=2)
    plot_coefficients(coeff_matrices[index], lib_names, fig)
    plt.title('Coefficient matrix')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95,
                        bottom=0.095,
                        left=0.125,
                        right=0.9,
                        hspace=0.6,
                        wspace=0.395)
    # Show the plot
    fig.suptitle("Alpha: " + str(alpha) + "Noise level: " + str(noise_level))

    plt.show()