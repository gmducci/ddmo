# optimizer.py
import logging
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.config import Config
from pymoo.core.problem import Problem

from ddmo.formulation import Formulation
from ddmo.optimizer.objective_functions import (
    DEFAULT_OBJECTIVES,
    ObjectiveContext,
    ObjectiveFunction,
)
from ddmo.solver.result import SolverResult
from ddmo.utils.data_constructor import DataConstructor

logger = logging.getLogger(__name__)
# Configure environment
warnings.filterwarnings("ignore")
Config.warnings['not_compiled'] = False


@dataclass
class EvaluationResult:
    obj_func_score: dict[str, float]
    solution: Any | None
    regression_coefficients: Any | None
    opt_vars: np.ndarray
    symbolic_model: Any | None = None
    feature_matrix: Any | None = None


class Optimizer(Problem):
    def __init__(
        self,
        data_instance: DataConstructor,
        solver: Callable,
        objective_functions: Mapping[str, ObjectiveFunction] | None = None,
        formulation: Formulation | None = None,
        solver_param_names: list[str] | None = None,
        objectives: Mapping[str, ObjectiveFunction] | None = None,
    ):
        super().__init__(
            n_var=1, n_obj=None, n_ieq_constr=0, xl=0.0, xu=1.0, vtype=float
        )

        self.data_instance = data_instance
        self.solver = solver
        self.formulation = formulation
        if objective_functions is not None and objectives is not None:
            raise ValueError(
                "Pass either objective_functions or the legacy objectives argument, not both."
            )
        if objective_functions is not None:
            selected_objectives = objective_functions
        elif objectives is not None:
            selected_objectives = objectives
        else:
            selected_objectives = DEFAULT_OBJECTIVES
        if not selected_objectives:
            raise ValueError("At least one objective function is required.")
        if any(not callable(function) for function in selected_objectives.values()):
            raise TypeError("Every objective function must be callable.")

        self.objectives = dict(selected_objectives)
        self.n_obj = len(self.objectives)
        self.solver_param_names = solver_param_names if solver_param_names else []
        self.n_candidate_evaluations = 0
        self.n_failed_evaluations = 0

    def _compute_metrics(self, optimization_variables, *args):
        self.n_candidate_evaluations += 1
        # Create param_dict if mapi_label is available
        param_dict = {}
        if hasattr(self, "mapi_label"):
            param_dict = {
                name: optimization_variables[idx]
                for name, idx in self.mapi_label.items()
            }

        if self.formulation is not None:
            lhs = self.formulation.get_lhs(
                self.data_instance,
                optimization_variables,
                opt_param_dict=param_dict,
            )
            rhs = self.formulation.get_rhs(
                self.data_instance,
                optimization_variables,
                opt_param_dict=param_dict,
            )
            solver_args = (lhs, rhs)
            solver_args = (lhs, rhs)
        else:
            solver_args = (optimization_variables,)

        # Extract solver-specific parameters if any
        solver_kwargs = {}
        if self.solver_param_names:
            for name in self.solver_param_names:
                if name in param_dict:
                    solver_kwargs[name] = param_dict[name]

        solver_result: SolverResult = self.solver(*solver_args, **solver_kwargs)

        if solver_result is None or not solver_result.success:
            self.n_failed_evaluations += 1
            return EvaluationResult(
                obj_func_score={name: np.nan for name in self.objectives},
                solution=None,
                regression_coefficients=None,
                opt_vars=np.array([np.nan]),
                symbolic_model=None,
                feature_matrix=solver_result.feature_matrix if solver_result else None,
            )

        context = ObjectiveContext(
            data=self.data_instance,
            solver_result=solver_result,
        )
        objective_scores = {}
        objective_failed = False
        for name, obj_func in self.objectives.items():
            try:
                objective_scores[name] = float(obj_func(context))
            except Exception:
                logger.exception("Objective '%s' failed.", name)
                objective_scores[name] = 1e5
                objective_failed = True
        if objective_failed:
            self.n_failed_evaluations += 1

        logger.debug("Objective function values: %s", objective_scores)
        return EvaluationResult(
            obj_func_score=objective_scores,
            solution=solver_result.solution,
            regression_coefficients=solver_result.coefficients,
            opt_vars=solver_result.opt_vars,
            symbolic_model=solver_result.model,
            feature_matrix=solver_result.feature_matrix,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        flat_x = [tuple(row) for row in x]
        # Run model on each candidate
        results = [self._compute_metrics(xi) for xi in flat_x]

        # Get objective names for consistent ordering
        objective_names = list(self.objectives.keys())

        # Initialize containers with fallback values
        objective_values = []
        solutions = []
        coeff_matrices = []
        alphas = []
        models = []
        feature_matrices = []

        for res in results:
            if res is not None and res.obj_func_score is not None:
                scores = res.obj_func_score
                # Check if any objective is valid (not all NaN)
                if isinstance(scores, dict):
                    is_valid = not all(np.isnan(v) for v in scores.values())
                else:
                    is_valid = not np.all(np.isnan(scores))

                if is_valid:
                    obj = [scores.get(name, np.inf) for name in objective_names]
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
            else:
                obj = [np.inf] * self.n_obj
                sol = None
                coeff = None
                alpha = np.array([np.nan])
                sindy_model = None
                theta = None

            objective_values.append(obj)
            solutions.append(sol)
            coeff_matrices.append(coeff)
            alphas.append(alpha)
            models.append(sindy_model)
            feature_matrices.append(theta)

        # Convert to numpy array and stack objectives dynamically
        obj_arrays = np.array(objective_values, dtype=float)

        dw_name = next(
            (
                name
                for name in ("durbin_watson_deviation", "dw_coefficient")
                if name in objective_names
            ),
            None,
        )
        if dw_name is not None:
            dw_idx = objective_names.index(dw_name)
            dw_stats = obj_arrays[:, dw_idx]
            logger.debug(
                "Durbin-Watson deviation monitoring: worst value = %.4f",
                np.nanmax(dw_stats),
            )
            out["durbin_watson_deviation"] = dw_stats

        # Store output - dynamically stack all objective columns
        out["F"] = obj_arrays
        out["solution"] = np.asarray(solutions, dtype=object)
        out["coeff_matrix"] = np.asarray(coeff_matrices, dtype=object)
        out["alphas"] = np.asarray(alphas, dtype=object)
        out['models'] = models
        out['theta'] = np.asarray(feature_matrices, dtype=object)
