from pprint import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
from ddmo.utils.data_constructor import DataConstructor
from ddmo.utils.constructors import OptimizerSettings
from ddmo.solver.settings import SolverSettings
from ddmo.formulation.formulation_settings import FormulationSettings
from ddmo.library_functions import LibraryBuilder
from ddmo import BENCHMARK_FOLDER, identify
import warnings

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
        .add_polynomial(max_degree=2)                # x, x², x*y
        .add_hill(exponent_param="gamma_1")          # 1/(1+x^γ₁)
        .add_exponential(rate_param="gamma_2")       # exp(-γ₂·x)
    )
    
    x_0 = [x_0_train]

    # Create a DataConstructor instance
    skip = 2
    training_data = DataConstructor(
        x_train=x_raw[::skip, :],
        time_array=time_array[::skip],
        x_dot=x_dot_hat[::skip, :],
        library_object=library.build,  # Pass the build method
        x_0=x_0,
        x_raw=x_raw[::skip, :]
    )
    
    # Library parameters (for SINDy function library) - tuple format: (min, max)
    library_bounds = {
        "gamma_1": (8.4, 9.6),
        "gamma_2": (1.0, 10.0),
    }

    # Formulation parameters (for weak formulation)
    formulation_bounds = {
        "stride": (5, 120),
        "window_length": (51, 401),
    }

    # Unified optimizer settings with separated bounds
    optimizer_settings = OptimizerSettings(
        library_bounds=library_bounds,
        formulation_bounds=formulation_bounds,
        n_obj=2,
        n_gen=10,
        algorithm="NSGA2",
        pop_size=50
    )

    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "alpha": 0.2, 
            "threshold": 0.25,
        }
    )
    
    formulation_settings = FormulationSettings(
        formulation="weak",
        test_function="PolynomialWindow",
        degree=3,
        window_length=51,
        stride=5
    )

    # Run identification
    result = identify(
        data_instance=training_data,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings,
    )

    # Access results via named attributes
    print(result.alphas)

    plt.plot(result.solutions[0])
    plt.plot(x_raw[::skip], alpha=0.5)
    plt.show()