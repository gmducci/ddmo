from ..feature_library.weak_pde_library import WeakPDELibrary
from .optimizer import Optimizer
import numpy as np
import pysindy as ps
from sklearn.preprocessing import normalize, StandardScaler, RobustScaler, QuantileTransformer
from scipy.integrate import odeint
from sklearn import linear_model, metrics
import pymoo.gradient.toolbox as anp
from sklearn.linear_model import Lasso
from pysindy.optimizers import STLSQ, SR3, ConstrainedSR3, EnsembleOptimizer

class Pysindy_Solver(Optimizer):
    def __init__(self,
                 x_train,
                 time_array,
                 x_dot,
                 library_object,
                 x_0,
                 x_raw,
                 optimizer="stlsq",
                 solver="lasso",
                 solver_parameters={},
                 scale_theta=False,
                 objectives=None):
        # forward everything to the parent
        super().__init__(
            x_train=x_train,
            time_array=time_array,
            x_dot=x_dot,
            library_object=library_object,
            x_0=x_0,
            x_raw=x_raw,
            optimizer=optimizer,
            solver_parameters = solver_parameters,
            solver=solver,
            scale_theta=scale_theta,
            objectives=objectives,
        )


    def _compute_metrics(self, x_row):
        result = self.solver(x_row)
        if result[0] is None:
            return [[np.nan, np.nan, np.nan, np.nan], []]

        sol, regression_coeff, opt_vars, model = self.solver(x_row)
        objective_funcs = []
        for obj_func in self.objectives:
            try:
                objective_funcs.append(obj_func(self, sol))
            except Exception:
                objective_funcs.append(1e5)

        print(f"Objective function values: {objective_funcs}")
        return objective_funcs, sol, regression_coeff, opt_vars, model

    

    def solver(self, optimization_vars):
        """
        """
        if len(optimization_vars) == 1:
            opt_vars, = optimization_vars  # Unpack if it's a 1D tuple
        else:
            opt_vars = optimization_vars  # Leave it as is otherwise

        # Optimizer dispatch logic
        _optimizers = {
            "STLSQ": STLSQ,
            "SR3": SR3,
            "ConstrainedSR3": ConstrainedSR3,
            "EnsembleOptimizer": EnsembleOptimizer,
            "Lasso": Lasso,
        }

        self.alphas.append(opt_vars)
        if np.isscalar(opt_vars) or isinstance(opt_vars, np.float64):
            opt_vars = (opt_vars,)
        
        named_funcs = self.library_object([*opt_vars])
        functions = [f for f, _ in named_funcs]
        function_names = [name for _, name in named_funcs]

        lib_custom = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names
        )
        lib_custom.fit(self.x_train)
        dt = self.time_array[1] - self.time_array[0]
        params_dict = self.solver_parameters[0]
        optimizer = _optimizers[self.optimizer](**params_dict)

          # Increase threshold to enforce more sparsity
        try:

            model = ps.SINDy(feature_library=lib_custom, optimizer=optimizer)
            if isinstance(self.x_train[0], (list, tuple, np.ndarray)):
                X = np.vstack(self.x_train)
            else:
                X = self.x_train
            model.fit(X, t=dt, x_dot=self.x_dot)

            regression_coeff = model.coefficients()
            regression_coeff = regression_coeff.T
            sol_list = []

            for i in range(self.number_initial_conditions):

                sol_ = model.simulate(self.initial_values[i], self.time_array)

                sol_list.append(sol_)
            sol = np.vstack(sol_list)

            x = sol

            return x, regression_coeff, opt_vars, model
        except Exception as e:
            print(f"Error during solver execution: {e}")
            return None, None, None, None, None

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

        for res in results:
            if (
                isinstance(res, (list, tuple)) and
                res[0] is not None and
                not np.all(np.isnan(res[0]))
            ):
                obj, sol, coeff, alpha, sindy_model = res
            else:
                obj = [np.inf, np.inf]
                sol = None
                coeff = None
                alpha = np.array([np.nan])
                sindy_model = None


            objectives.append(obj)
            solutions.append(sol)
            coeff_matrices.append(coeff)
            alphas.append(alpha)
            models.append(sindy_model)

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