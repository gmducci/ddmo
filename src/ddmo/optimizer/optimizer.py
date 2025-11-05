# optimizer.py
import multiprocessing as mp
import warnings

import numpy as np
import pymoo.gradient.toolbox as anp
from pymoo.config import Config
from pymoo.core.problem import Problem

from dataclasses import dataclass
from typing import Any, Optional

from ddmo.formulation import Formulation
from ddmo.optimizer.objective_functions import (
    compute_dw_coefficient,
    compute_mse,
    compute_ppcc_matrix,
)
from ddmo.utils.data_constructor import DataConstructor
from typing import Callable

# Configure environment
warnings.filterwarnings("ignore")
Config.warnings['not_compiled'] = False
mp.set_start_method("fork", force=True)

@dataclass
class EvaluationResult:
    obj_func_score: list[float]
    solution: Optional[Any]
    regression_coefficients: Optional[Any]
    opt_vars: np.ndarray
    symbolic_model: Optional[Any] = None
    feature_matrix: Optional[Any] = None

class Optimizer(Problem):
    def __init__(self,
                 data_instance: DataConstructor,
                 # library_factory: Callable[[list[float]], list[NamedFunc]],
                 # formution: Formulation,
                 solver: Callable,
                 objectives=None,
                 objective_functions=None,
                 formulation: Formulation | None = None):

        super().__init__(n_var=1,
                         n_obj=None,
                         n_ieq_constr=0,
                         xl=0.,
                         xu=1.,
                         vtype=float)
        
        self.data_instance = data_instance
        self.solver = solver
        self.formulation = formulation
        #self.library_factory = library_factory
        # self.formution = formution
        self.objectives = objectives if objectives is not None else [compute_mse, compute_dw_coefficient]
        self.objective_functions = objective_functions
        self.n_obj = len(self.objectives)

        self.coefficient_matrix = []
        self.alphas = []
        self.results = []
        self.solution = []

    def _compute_metrics(self, optimization_variables, *args):

        _objective_functions_dict = {
            "mse": compute_mse,
            "dw_coefficient": compute_dw_coefficient,
            "ppcc_matrix": compute_ppcc_matrix
        }

        if self.formulation is not None:
            lhs = self.formulation.get_lhs(self.data_instance, optimization_variables)
            rhs = self.formulation.get_rhs(self.data_instance, optimization_variables)
            solver_args = (lhs, rhs)
        else:
            solver_args = (optimization_variables,)

        solver_output = self.solver(*solver_args)

        if solver_output is None:
            return EvaluationResult(
                obj_func_score=[np.nan] * self.n_obj,
                solution=None,
                regression_coefficients=None,
                opt_vars=np.array([np.nan]),
                symbolic_model=None,
                feature_matrix=None,
            )

        feature_matrix = None
        try:
            sol, regression_coeff, opt_vars, model, feature_matrix = solver_output
        except ValueError:
            sol, regression_coeff, opt_vars, model = solver_output
            feature_matrix = None

        if sol is None:
            return EvaluationResult(
                obj_func_score=[np.nan] * self.n_obj,
                solution=None,
                regression_coefficients=None,
                opt_vars=np.array([np.nan]),
                symbolic_model=None,
                feature_matrix=feature_matrix,
            )

        objective_funcs = []
        for obj_func in self.objectives:
            try:
                objective_funcs.append(obj_func(self.data_instance, sol))
            except Exception:
                objective_funcs.append(1e5)

        print(f"Objective function values: {objective_funcs}")
        return EvaluationResult(
            obj_func_score=objective_funcs,
            solution=sol,
            regression_coefficients=regression_coeff,
            opt_vars=opt_vars,
            symbolic_model=model,
            feature_matrix=feature_matrix,
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Clear previous state
        self.coefficient_matrix.clear()
        self.alphas.clear()
        self.results.clear()
        self.solution.clear()

        flat_x = [tuple(row) for row in x]
        # Run model on each candidate
        results = [self._compute_metrics(xi) for xi in flat_x]

        # Initialize containers with fallback values
        objectives = []
        solutions = []
        coeff_matrices = []
        alphas = []
        models = []
        feature_matrices = []

        for res in results:
            if (
                res is not None and
                res.obj_func_score is not None and
                not np.all(np.isnan(res.obj_func_score))
            ):
                obj = res.obj_func_score
                sol = res.solution
                coeff = res.regression_coefficients
                alpha = res.opt_vars
                sindy_model = res.symbolic_model
                theta = res.feature_matrix
            else:
                obj = [np.inf] * self.n_obj
                sol = None
                coeff = None
                alpha = np.array([np.nan])
                sindy_model = None
                theta = None

            objectives.append(obj)
            solutions.append(sol)
            coeff_matrices.append(coeff)
            alphas.append(alpha)
            models.append(sindy_model)
            feature_matrices.append(theta)

        # Convert to numpy arrays
        obj_arrays = np.array(objectives, dtype=float)
        dw_stats = obj_arrays[:, 1]
        max_dw_stats = 2 - np.nanmin(dw_stats)

        # Monitoring output
        print(
            f"+{'-' * 60}+\n"
            f"| DW statistic monitoring: Maximum value = {max_dw_stats:.4f} |\n"
            f"+{'-' * 60}+"
        )

        # Store output
        out["F"] = anp.column_stack([obj_arrays[:, 0], dw_stats])
        out["solution"] = np.asarray(solutions, dtype=object)
        out["coeff_matrix"] =  np.asarray(coeff_matrices, dtype=object)
        out["alphas"] = np.asarray(alphas, dtype=object)
        out["dw_coeff"] = dw_stats
        out['models'] = models
        out['theta'] = np.asarray(feature_matrices, dtype=object)
