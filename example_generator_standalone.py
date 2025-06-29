import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from ddmo.equations import goodwin_eqs
import ddmo.differentiation.differentiation as diff
import os
from ddmo.optimizer.optimizer import Optimizer


benchmark_folder = '/home/ducci/PROJECT/ddmo-main/src/ddmo/dataset_benchmark'
x_0_train = np.load(benchmark_folder + '/x_0.npy')
x_raw = np.load(benchmark_folder + '/x_train_noise.npy')
x_train= np.load(benchmark_folder + '/x_train.npy')
x_dot_hat = np.load(benchmark_folder + '/x_dot_train.npy') 
time_array = np.load(benchmark_folder + '/time.npy')   
x_dot_gt = np.load(benchmark_folder + '/x_dot_gt.npy')

from ddmo import identify
from ddmo.library_functions import functions_template 
from typing import Callable, List, Tuple

NamedFunc = Tuple[Callable[[float, float], float], str]
def get_base_functions(params: List[float]) -> List[NamedFunc]:
    alpha = params
    return [
        (lambda x: x, "x"),
        (lambda x: (1 / (1 + x**alpha)), f"1 / 1 + x^{alpha}")
    ]

problem, res, obj_space, solutions, coeff_matrices, alphas = identify(
    x_train = x_train[40:-40,:],
    x_raw=x_raw[40:-40,:],
    x_dot=x_dot_hat[40:-40,:],
    t = time_array,
    x0=x_0_train,
    library_object=get_base_functions,
    bounds= [np.array([5]), np.array([10])],
    weak_form=False,
    scale_theta=True,
    n_vars=1,
    optimizer='STLSQ',
    solver_parameters={
        "threshold": 0.1},
    solver='lasso',
    number_generations=5,
    population_size=25)

print(alphas)