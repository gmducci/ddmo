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

if __name__ == "__main__":
    # Load benchmark data
    x_0_train = np.load(BENCHMARK_FOLDER / 'x_0.npy')
    x_raw = np.load(BENCHMARK_FOLDER / 'x_train_noise.npy')
    x_train = np.load(BENCHMARK_FOLDER / 'x_train.npy')
    x_dot_hat = np.load(BENCHMARK_FOLDER / 'x_dot_train.npy')
    time_array = np.load(BENCHMARK_FOLDER / 'time.npy')

    # Define the function library using the fluent API
    library = (
        LibraryBuilder()
        .add_polynomial(max_degree=2, include_interactions=True)  # x, x², x*y
        .add_hill(exponent_param="gamma_1")  # 1/(1+x^γ₁)
        .add_exponential(rate_param="gamma_2")  # exp(-γ₂·x)
    )

    x_0 = [x_0_train]
    # Create a DataConstructor instance
    skip = 5
    training_data = DataConstructor(
        x_train=x_train[20:-20, :],
        time_array=time_array[20:-20],
        x_dot=x_dot_hat[20:-20, :],
        x_0=x_0,
        x_raw=x_raw[20:-20, :],
    )

    # Library parameters (for SINDy function library) - tuple format: (min, max)
    library_bounds = {
        "gamma_1": (8.4, 9.6),
        "gamma_2": (1.0, 10.0),
    }

    # Formulation parameters (for weak formulation)
    formulation_bounds = {}

    # Solver bounds: Optimize the SINDy threshold!
    solver_bounds = {"threshold": (0.01, 1.0)}

    # Unified optimizer settings with separated bounds
    optimizer_settings = OptimizerSettings(
        library_bounds=library_bounds,
        formulation_bounds=formulation_bounds,
        solver_bounds=solver_bounds,
        n_gen=3,
        algorithm="NSGA3",
        pop_size=50,
    )

    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "normalize_columns": False,
            "alpha": 0.35,
            "threshold": 0.45,
        },
        scale_features=True,
    )

    formulation_settings = FormulationSettings(
        formulation="strong",
    )

    # Run identification
    result = identify(
        data_instance=training_data,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings,
        library_object=library.build,
    )

    # Access results via named attributes
    print(result.alphas)

    plt.plot(result.solutions[0])
    plt.plot(x_raw, alpha=0.5)
    plt.show()
