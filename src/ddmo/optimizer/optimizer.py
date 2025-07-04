# optimizer.py
import warnings

import numpy as np
import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem

# import statsmodels.api as sm
from .objective_functions import (
    compute_dw_coefficient,
    compute_mse,
)
from .solver import solver

warnings.filterwarnings("ignore")
import multiprocessing as mp

from pymoo.config import Config

Config.warnings['not_compiled'] = False
mp.set_start_method("fork", force=True)

class Optimizer(Problem):
    def __init__(self,
                 x_train,
                 time_array,
                 x_dot,
                 library_object,
                 x_0,
                 x_raw,
                 optimizer="STLSQ",
                 solver="lasso",
                 solver_parameters={},
                 scale_theta=False,
                 objectives=None,
                 objective_functions=None):

        """Initialize the class with sparse algorithm variables.

        Args:
            x_train_noise (list): Training data with noise. Or pre-smoothed data.
            time_array (list): time array related to the training data.
            x_dot (list): numerical derivative of training data. This corresponds to the LHS of the system
        """

        super().__init__(n_var=1,
                        n_obj= None,
                        n_ieq_constr=0,
                        xl = 0.,
                        xu = 1., vtype=float)

        self.x_train = x_train
        self.time_array = time_array
        self.initial_values = x_0
        self.x_dot = x_dot
        self.library_object = library_object
        self.x_raw = x_raw
        self.model_type = solver
        self.optimizer = optimizer
        self.solver_parameters = solver_parameters,
        self.scale_theta = scale_theta
        self.dimension = len(x_train[0])
        self.number_initial_conditions = len(x_0)
        self.objectives = objectives if objectives is not None else [compute_mse, compute_mse]
        self.objective_functions = objective_functions,
        self.n_obj = len(self.objectives)

        self.coefficient_matrix = []
        self.alphas = []
        self.results = []
        self.solution = []

    def _compute_metrics(self, x_row, regression):

        sol, x_ref, regression_coeff, opt_vars = solver(self, x_row, regression)
        objective_funcs = []
        for obj_func in self.objectives:
            try:
                objective_funcs.append(obj_func(self, sol))
            except Exception:
                objective_funcs.append(1e5)

        print(f"Objective function values: {objective_funcs}")
        return objective_funcs, sol, regression_coeff, opt_vars

    def _evaluate(self, x, out, *args, **kwargs):
        self.coefficient_matrix.clear()
        self.alphas.clear()
        self.results.clear()
        self.solution.clear()

        coefficient_matrix = []
        coefficient_matrix.clear()
        flat_x = [tuple(row) for row in x]
        f = [self._compute_metrics(xi, regression) for xi in flat_x]


        objectives = [fi[0] for fi in f]
        data = [row if row else [np.nan, np.nan] for row in objectives]
        obj_arrays = np.array(data)
        solution = [fi[1] for fi in f]
        coeff_matrix = [fi[2] for fi in f]
        alpha = [fi[3] for fi in f]


        dw_stats = np.asarray(obj_arrays[:, 1])

        max_dw_stats = 2 - min(dw_stats)

        # Monitoring print message
        message = (
            f"+{'-' * 60}+\n"
            f"| DW statistic monitoring: Maximum value = {max_dw_stats:.4f} |\n"
            f"+{'-' * 60}+"
        )

        print(message)
        names = []
        funct_d = []

        # for alpha_ in np.asarray(alpha):

        #     names.append(np.asarray(self.library_object._functions(alpha_)[1]))

        #     funct_d.append(np.asarray(self.library_object._functions(alpha_)[2]))

        out["F"] = anp.column_stack([obj_arrays[:, 0],  obj_arrays[:, 1]])

        solution = np.asanyarray(solution)
        lasso_coeff = np.asanyarray(coeff_matrix)
        out.__setitem__("solution", solution)
        out.__setitem__("coeff_matrix", lasso_coeff)
        out.__setitem__("alphas", np.asarray(alpha))
        out.__setitem__("dw_coeff", dw_stats)
