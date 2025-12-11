from collections.abc import Callable
from pprint import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
from ddmo.utils.data_constructor import DataConstructor
from ddmo.utils.constructors import OptimizerSettings
from ddmo.solver.settings import SolverSettings
from ddmo.formulation.formulation_settings import FormulationSettings
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

    NamedFunc = tuple[Callable[[float, float], float], str]

    # Define the library of functions
    def get_base_functions(params: list[float]) -> list[NamedFunc]:
        gamma_1, gamma_2 = params[0:2]
        return [
            (lambda x: x, lambda x: x),
            (lambda x: x**2, lambda x: f"{x}^2"),
            (lambda x: (1 / (1 + x**gamma_1)), lambda x: f"1 / 1 + {x}^{gamma_1}"),
            (lambda x: np.exp(- gamma_2 * x), lambda x: f"exp( {-gamma_2} {x} )")
        ]

    x_0 = [x_0_train]

    # Create a DataConstructor instance
    skip = 1
    training_data = DataConstructor(
        x_train=x_train[::skip, :],
        time_array=time_array[::skip],
        x_dot=x_dot_hat[::skip, :],
        library_object=get_base_functions,
        x_0=x_0,
        x_raw=x_raw[::skip, :]
    )
    
    # Define optimization bounds
    lower_bounds = {
        "gamma_1": 8.4,
        "gamma_2": 1.0,
        "stride": 5,
        "degree": 3,
        "window_length": 51,
    }

    upper_bounds = {
        "gamma_1": 9.6,
        "gamma_2": 10.0,
        "stride": 120,
        "degree": 7,
        "window_length": 401,
    }

    # Unified optimizer settings - simple and clean!
    optimizer_settings = OptimizerSettings(
        xl=lower_bounds,
        xu=upper_bounds,
        n_obj=2,
        n_gen=10,
        algorithm="NSGA2",  # Options: "NSGA2", "NSGA3"
        pop_size=50
    )

    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "alpha": 0.15, 
            "threshold": 0.2,
        }
    )
    
    formulation_settings = FormulationSettings(
        formulation_name="weak",
        formulation_attributes={
            "test_functions_class": "PolynomialWindow",
            "degree": 3,  
            "window_length": 51,
            "stride": 5           
        }   
    )

    # Run identification
    identification = identify(
        data_instance=training_data,
        optimizer_settings=optimizer_settings,
        solver_instance=solver_settings,
        formulation_instance=formulation_settings,
    )

    (problem,
        res,
        obj_space,
        solutions,
        coeff_matrices,
        alphas,
        sindy_model,
        theta_matrices,
    ) = identification

    print(alphas)

    plt.plot(solutions[0])
    plt.plot(x_raw[::skip], alpha=0.5)
    plt.show()