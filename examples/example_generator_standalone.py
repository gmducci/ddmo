from collections.abc import Callable
from pprint import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
from ddmo.utils.data_constructor import DataConstructor
from ddmo.utils.constructors import (
    ProblemSettings,
    AlgorithmSettings,
    TerminationSettings)
from ddmo.solver.registry import SOLVER_REGISTRY
from ddmo.solver.settings import SolverSettings
from ddmo.formulation.formulation_settings import FormulationSettings
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.integrate import solve_ivp
from ddmo.formulation.registry import FORMULATION_REGISTRY
from ddmo import BENCHMARK_FOLDER, identify
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    x_0_train = np.load(BENCHMARK_FOLDER / 'x_0.npy')
    x_raw = np.load(BENCHMARK_FOLDER / 'x_train_noise.npy')
    x_train= np.load(BENCHMARK_FOLDER / 'x_train.npy')
    x_dot_hat = np.load(BENCHMARK_FOLDER / 'x_dot_train.npy')
    time_array = np.load(BENCHMARK_FOLDER / 'time.npy')
    x_dot_gt = np.load(BENCHMARK_FOLDER / 'x_dot_gt.npy')


    NamedFunc = tuple[Callable[[float, float], float], str]

    # Define here the library of functions
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
        x_train=x_train[::skip, :], # Skipping data every 5 points -> downsampled data
        time_array=time_array[::skip],
        x_dot=x_dot_hat[::skip, :],
        library_object=get_base_functions,
        x_0=x_0,
        x_raw=x_raw[::skip, :])
    
    # Definition of opt boundaries of the optimization variables
    # --- 1) library parameters (used by get_base_functions) --------------
    library_lower_bounds = {
        "gamma_1": 8.4,
        "gamma_2": 1.0,
    }

    library_upper_bounds = {
        "gamma_1": 9.6,
        "gamma_2": 10.0,
    }

    # --- 2) formulation / test-function parameters -----------------------
    formulation_lower_bounds = {
        "stride": 5,
        "degree": 3,
        "window_length": 51,
    }

    formulation_upper_bounds = {
        "stride": 120,
        "degree": 7,
        "window_length": 401,
    }

    library_param_names = list(library_lower_bounds.keys())
    formulation_param_names = list(formulation_lower_bounds.keys())

    # --- 3) combined bounds passed to ProblemSettings (backward compatible)
    lower_bounds = {**library_lower_bounds, **formulation_lower_bounds}
    upper_bounds = {**library_upper_bounds, **formulation_upper_bounds}
    problem_settings = ProblemSettings(n_var=len(lower_bounds),
                                       n_obj=2, 
                                       xl_dict= lower_bounds,
                                       xu_dict= upper_bounds)
    
    algorithm_settings = AlgorithmSettings(pop_size=50,
                                           sampling=FloatRandomSampling(),   # default
                                           crossover=SBX(eta=15, prob=0.9),  # default
                                           mutation=PM(eta=20))              # default
    
    termination_settings = TerminationSettings(n_gen=10)

    solver_settings = SolverSettings(solver_name="pysindy",
                                    regularization="STLSQ",
                                    regularization_parameters={
                                        "alpha": 0.15, 
                                        "threshold": 0.2,
                                    },
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

    identification = identify(
        data_instance=training_data,
        problem_settings=problem_settings,
        algorithm_settings=algorithm_settings,
        termination_settings=termination_settings,
        solver_instance=solver_settings,
        formulation_instance=formulation_settings,
        seed=None,
        verbose=True
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
    plt.plot(x_raw[::skip], alpha = 0.5)
    plt.show()