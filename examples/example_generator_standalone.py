from pathlib import Path
from pprint import pprint as pp
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from scipy.integrate import odeint

import ddmo.differentiation.differentiation as diff
from ddmo import BENCHMARK_FOLDER, identify
from ddmo.equations import goodwin_eqs
from ddmo.library_functions import functions_template
from ddmo.optimizer.optimizer import Optimizer

if __name__ == "__main__":
    x_0_train = np.load(BENCHMARK_FOLDER / 'x_0.npy')
    x_raw = np.load(BENCHMARK_FOLDER / 'x_train_noise.npy')
    x_train= np.load(BENCHMARK_FOLDER / 'x_train.npy')
    x_dot_hat = np.load(BENCHMARK_FOLDER / 'x_dot_train.npy') 
    time_array = np.load(BENCHMARK_FOLDER / 'time.npy')   
    x_dot_gt = np.load(BENCHMARK_FOLDER / 'x_dot_gt.npy')

    NamedFunc = Tuple[Callable[[float, float], float], str]
    def get_base_functions(params: List[float]) -> List[NamedFunc]:
        alpha, beta = params
        return [
            (lambda x: x, lambda x: x),
            (lambda x: x**2, lambda x: f"{x}^2"),
            (lambda x: (1 / (1 + x**alpha)), lambda x: f"1 / 1 + {x}^{alpha}"),
            (lambda x: np.exp(- beta * x), lambda x: f"exp( {-beta} {x} )")
        ]

    x_0 = [x_0_train]
    problem, res, obj_space, solutions, coeff_matrices, alphas, sindy_model = identify(
        x_train = x_train[40:-40,:],
        x_raw=x_raw[40:-40,:],
        x_dot=x_dot_hat[40:-40,:],
        t = time_array,
        x0=x_0,
        library_object=get_base_functions,
        bounds= [np.array([1, 1]), np.array([10, 10])],
        weak_form=False,
        scale_theta=True,
        n_vars=2,
        optimizer='STLSQ',
        solver_parameters={
            "threshold": 0.001},
        solver='lasso',
        number_generations=5,
        population_size=25)

    pp(alphas)
