from collections.abc import Callable
from pprint import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
from ddmo.utils.data_constructor import DataConstructor
from ddmo.utils.constructors import (
    ProblemSettings,
    AlgorithmSettings,
    TerminationSettings,
    GASettings)
from ddmo.solver.registry import SOLVER_REGISTRY
from ddmo.solver.settings import SolverSettings
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

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
    def get_base_functions(params: list[float]) -> list[NamedFunc]:
        alpha, beta = params
        return [
            (lambda x: x, lambda x: x),
            (lambda x: x**2, lambda x: f"{x}^2"),
            (lambda x: (1 / (1 + x**alpha)), lambda x: f"1 / 1 + {x}^{alpha}"),
            (lambda x: np.exp(- beta * x), lambda x: f"exp( {-beta} {x} )")
        ]

    x_0 = [x_0_train]

    # Create a DataConstructor instance
    training_data = DataConstructor(
        x_train=x_raw[40:-40, :],
        time_array=time_array,
        x_dot=x_dot_hat[40:-40, :],
        library_object=get_base_functions,
        x_0=x_0,
        x_raw=x_raw)
    
    # Define all the optimization settings in a constructor
    problem_settings = ProblemSettings(n_var=2,
                                       n_obj=2, 
                                       xl=np.array([8.4, 1.0]),
                                       xu=np.array([9.6, 10.0]))
    
    algorithm_settings = AlgorithmSettings(pop_size=1000,
                                           sampling=FloatRandomSampling(),   # default
                                           crossover=SBX(eta=15, prob=0.9),  # default
                                           mutation=PM(eta=20))              # default
    
    termination_settings = TerminationSettings(n_gen=10)

    ga_settings = GASettings(problem=problem_settings,
                             algorithm=algorithm_settings,
                             termination=termination_settings, 
                             seed=None,
                             verbose=True)


    solver_settings = SolverSettings(solver_name="pysindy",
                                    optimizer="STLSQ",
                                    solver_parameters={
                                    "alpha": 0.01,
                                    "threshold": 0.1
                                    })

    solver_class = SOLVER_REGISTRY[solver_settings.solver_name]

    solver_settings = solver_class(data_instance=training_data,
                          optimizer=solver_settings.optimizer,
                          solver_parameters=solver_settings.solver_parameters)

    identification = identify(data_instance=training_data,
                              formulation="weak",
                              formulation_options={"time_array": time_array,
                                "x_train": x_train[40:-40, :],
                                  "window_length": 501,
                                  "stride": 200
                              },
                              ga_settings=ga_settings,
                              solver_instance=solver_settings)
    
    (
        problem,
        res,
        obj_space,
        solutions,
        coeff_matrices,
        alphas,
        sindy_model,
        theta_matrices,
    ) = identification

    print(alphas)
