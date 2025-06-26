from ..feature_library.weak_pde_library import WeakPDELibrary
from .optimizer import Optimizer
import numpy as np
import pysindy as ps
from sklearn.preprocessing import normalize, StandardScaler, RobustScaler, QuantileTransformer
from scipy.integrate import odeint
from sklearn import linear_model, metrics
import pymoo.gradient.toolbox as anp


class Pysindy_Solver(Optimizer):
    def __init__(self,
                 x_train,
                 time_array,
                 x_dot,
                 library_object,
                 x_0,
                 x_raw,
                 solver="lasso",
                 solver_param={},
                 alpha_lasso=1e-4,
                 target="trajectory",
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
            solver=solver,
            alpha_lasso=alpha_lasso,
            target=target,
            scale_theta=scale_theta,
            objectives=objectives,
        )

        self.regression_model = {
            'lasso': linear_model.Lasso(**solver_param)
        }[solver]

    def _compute_metrics(self, x_row, regression):

        sol, x_ref, regression_coeff, opt_vars = self.solver(x_row, regression)
        objective_funcs = []
        for obj_func in self.objectives:
            try:
                objective_funcs.append(obj_func(self, sol))
            except Exception:
                objective_funcs.append(1e5)

        print(f"Objective function values: {objective_funcs}")
        return objective_funcs, sol, regression_coeff, opt_vars

        objective_funcs = []
        for obj_func in self.objectives:
            try:
                objective_funcs.append(obj_func(self, sol))
            except Exception:
                objective_funcs.append(1e5)

        print(f"Objective function values: {objective_funcs}")
        return objective_funcs, sol, regression_coeff, opt_vars

    def solver(self, optimization_vars, regression):
        """
        """
        if len(optimization_vars) == 1:
            opt_vars, = optimization_vars  # Unpack if it's a 1D tuple
        else:
            opt_vars = optimization_vars  # Leave it as is otherwise

        self.alphas.append(opt_vars)
        self.library_object.set_base_funcs((opt_vars[0], opt_vars[1]))

        translator = FunctionLibraryTranslator(self.library_object)
        sindy_library = translator.to_sindy_library(alpha)

        model = ps.SINDy(feature_library=sindy_library)
        model.fit(X, t=dt, x_dot=dX)

        regression_coeff = model.coefficients()
        regression_coeff = regression_coeff.T
        sol_list = []
        def get_rhs_local(ssp, t):
            ode_lib.fit(ssp)
            theta = ode_lib.transform_simple(ssp)
            rhs = theta @ regression_coeff
            return rhs

        for x_0_ in self.initial_values:

            sol_ = odeint(get_rhs_local, x_0_, self.time_array)

            sol_list.append(sol_)
        sol = np.vstack(sol_list)

        if self.target == 'trajectory':
            x_ref = self.x_raw
            x = sol

        elif self.target == 'derivative':
            x_ref = self.x_dot
            rhs = theta @ coeff_matrix
            x = rhs
        else:
            raise ValueError("Invalid target specified. Must be 'trajectory' or 'derivative'.")
        
        return x, x_ref, regression_coeff, opt_vars

    def _evaluate(self, x, out, *args, **kwargs):
        self.coefficient_matrix.clear()
        self.alphas.clear()
        self.results.clear()
        self.solution.clear()

        coefficient_matrix = []
        coefficient_matrix.clear()
        flat_x = [tuple(row) for row in x]
        f = [self._compute_metrics(xi, self.regression_model) for xi in flat_x]


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
