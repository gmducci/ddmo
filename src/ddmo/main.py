#!/usr/bin/env python3
"""
DDMO: One-stop sparse identification entry point

Usage:

  # As a script / CLI:
  $ python DDMO.py --ic "1,1,0;0.5,0.8,0" --n-gen 100 --pop-size 30 \
       --regression-model ridge --alpha-lasso 0.05 --scale-theta --target derivative \
       --weak-form --seed 123 --families poly,nonlinear --individual-funcs cos,sin

  # As a library:
  >>> from DDMO import identify
  >>> problem, res = identify(
  ...     x=x_array,
  ...     t=time_array,
  ...     x0=x0_array,
  ...     bounds=((1.0,4.5),(1.0,4.5)),
  ...     pop_size=50,
  ...     n_gen=100,
  ...     regression_model="lasso",
  ...     alpha_lasso=0.01,
  ...     scale_theta=True,
  ...     target="trajectory",
  ...     weak_form=False,
  ...     seed=42,
  ...     library_families=["poly","custom"],
  ...     individual_funcs=[lambda x: x**3]
  ... )
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, List, Any

import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import pysindy as ps

import ddmo.differentiation.differentiation as diff
# from ddmo.optimizer.optimizer import Optimizer
# from ddmo.optimizer.Scipy_Solver import Scipy_Solver
from ddmo.equations import plugflow_equation, lorenz, goodwin_eqs
import ddmo.library_functions.functions as fcs

import functools
import operator
#!/usr/bin/env python3
# ddmo/library_factory.py

from typing import Callable, List, Optional, Tuple, Any, Dict
import pysindy as ps
import ddmo.library_functions.functions as fcs  # for LibrarySindy / LibraryFunctions
import itertools
from typing import Callable, List, Tuple

# Type aliases for clarity
Func = Callable[..., float]
NamedFunc = Tuple[Func, str]


from typing import Callable, Tuple, List
import numpy as np

NamedFunc = Tuple[Callable[..., float], str]

# -----------------------------------------------------------------------------# Main identification function
# -----------------------------------------------------------------------------


def identify(
    x_train: np.ndarray, # it can be equal to x, but we can also use a smoothed version
    x_raw: np.ndarray, # raw data, used for initial conditions
    x_dot: np.ndarray,
    t: np.ndarray,
    x0:np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    n_objectives: int = 2, # number of objective functions
    n_vars: int = 2, # number of optimization variables
    population_size: int = 20,
    number_generations: int = 10,
    optimizer: str = "stlsq",  # 'pysindy', 'pysindy_weak', 'scikit'
    solver: str = "lasso",
    solver_parameters: dict = {},
    scale_theta: bool = True,
    weak_form: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
    library_object: Optional[Callable[[List[float]], Any]] = None,
):
    """
    Run sparse identification on provided data x(t).

    Mandatory:
      x: array of shape (n_traces, n_steps, n_vars)
      t: array of shape (n_steps,)

    Optional keyword args as before, plus:
      - library_funcs: generator of a pysindy CustomLibrary given opt_vars
      - individual_funcs: explicit list of basis functions

    Returns:
      A SparseIdentification instance with attributes:
        - result: pymoo result object
        - pareto_front: objective values
        - solutions: optimized parameter sets
        - coeff_matrices, alphas, dw_stats
    """

    from ddmo.optimizer.Pysindy_Solver import Pysindy_Solver
    problem = Pysindy_Solver(     x_train,
                                t,
                                x_dot,
                                library_object,
                                x0,
                                x_raw = x_raw,
                                optimizer=optimizer,
                                solver=solver,
                                solver_parameters = solver_parameters,
                                scale_theta= scale_theta,
                                # solver = pysindy, pysindy_weak, scikit
                                objectives=None)


    '''
    problem = Optimizer(
        x_train,
                                t,
                                x_dot,
                                library_object,
                                x0,
                                x_raw = x_raw,
                                solver=solver,
                                alpha_lasso = alpha_lasso,
                                #alpha_lasso=0.000005,
                                target= target,
                                scale_theta= scale_theta,
                                # solver = pysindy, pysindy_weak, scikit
                                weak_form = weak_form, # to remove
                                objectives=None)
    '''

    problem.n_obj = n_objectives
    problem.n_var = n_vars
    if isinstance(x0[0], (list, tuple)):
        problem.dimension = len(x0[0])
    else:
        problem.dimension = len(x0)
    problem.xl = bounds[0]
    problem.xu = bounds[1]

    
    algorithm = NSGA2(pop_size=population_size)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', number_generations))
    
    obj_space = res.F
    solutions = res.opt.get("solution")
    coeff_matrices = res.opt.get("coeff_matrix")
    alphas = res.opt.get("alphas")

    return problem, res, obj_space, solutions, coeff_matrices, alphas