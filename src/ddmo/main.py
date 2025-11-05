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

from collections.abc import Callable
from typing import Any, Mapping

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from ddmo.utils.data_constructor import DataConstructor
from ddmo.utils.constructors import GASettings
from ddmo.solver.base import BaseSolver
from ddmo.formulation import Formulation, WeakFormulation, StrongFormulation

# from ddmo.optimizer.optimizer import Optimizer
# from ddmo.optimizer.Scipy_Solver import Scipy_Solver

#!/usr/bin/env python3
# ddmo/library_factory.py


# Type aliases for clarity
Func = Callable[..., float]
NamedFunc = tuple[Func, str]


from collections.abc import Callable

NamedFunc = tuple[Callable[..., float], str]

# -----------------------------------------------------------------------------# Main identification function
# -----------------------------------------------------------------------------


def _build_formulation(formulation: Formulation | str | None,
                       formulation_options: Mapping[str, Any] | None) -> Formulation | None:
    if formulation is None:
        return None
    if isinstance(formulation, Formulation):
        return formulation
    if isinstance(formulation, str):
        kind = formulation.lower()
        options = dict(formulation_options or {})
        if kind == "weak":
            return WeakFormulation(**options)
        if kind == "strong":
            return StrongFormulation(**options)
        msg = f"Unknown formulation: {formulation}"
        raise ValueError(msg)
    msg = f"Unsupported formulation type: {type(formulation)!r}"
    raise TypeError(msg)


def identify(data_instance: DataConstructor,
             ga_settings: GASettings,
             solver_instance: BaseSolver,
             objective_functions: dict = None,
             formulation: Formulation | str | None = None,
             formulation_options: Mapping[str, Any] | None = None):
    
    """
    Run sparse identification on provided data x(t).

    Mandatory:
      x: array of shape (n_traces, n_steps, n_vars)
      t: array of shape (n_steps,)

    Optional keyword args as before, plus:
      - library_funcs: generator of a pysindy CustomLibrary given opt_vars
      - individual_funcs: explicit list of basis functions
      - formulation: either a Formulation instance or "weak"/"strong"
      - formulation_options: keyword arguments forwarded when instantiating a formulation

    Returns:
      A SparseIdentification instance with attributes:
        - result: pymoo result object
        - pareto_front: objective values
        - solutions: optimized parameter sets
        - coeff_matrices, alphas, dw_stats
    """
    # ===== begin opt =====    
    from ddmo.optimizer.optimizer import Optimizer
    resolved_formulation = _build_formulation(formulation, formulation_options)
    problem = Optimizer(
        data_instance=data_instance,
        # library_factory=library_factory,
        # formution=formution,
        solver=solver_instance.solve,
        objectives=None,
        objective_functions=objective_functions,
        formulation=resolved_formulation
    )
    
    problem.n_var = ga_settings.problem.n_var
    problem.n_obj = ga_settings.problem.n_obj
    
    problem.xl = ga_settings.problem.xl
    problem.xu = ga_settings.problem.xu  
    from pymoo.util.ref_dirs import get_reference_directions

    algorithm = NSGA2(pop_size=ga_settings.algorithm.pop_size,
                      sampling=ga_settings.algorithm.sampling,
                      crossover=ga_settings.algorithm.crossover,
                      mutation=ga_settings.algorithm.mutation,
                      eliminate_duplicates=ga_settings.algorithm.eliminate_duplicates)
    
    res = minimize(
        problem,
        algorithm,
        ('n_gen', ga_settings.termination.n_gen),
        seed=ga_settings.seed,
        verbose=ga_settings.verbose)
    # ===== end opt =====

    obj_space = res.F
    solutions = res.opt.get("solution", to_numpy=False)
    coeff_matrices = res.opt.get("coeff_matrix", to_numpy=False)
    alphas = res.opt.get("alphas", to_numpy=False)
    model = res.opt.get("models", to_numpy=False)
    theta = res.opt.get("theta", to_numpy=False)

    return problem, res, obj_space, solutions, coeff_matrices, alphas, model, theta
