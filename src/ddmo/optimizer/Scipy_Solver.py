from ..feature_library.weak_pde_library import WeakPDELibrary
from .optimizer import Optimizer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from scipy.integrate import odeint
import pymoo.gradient.toolbox as anp

# Default hyperparameters for supported regressors
_DEFAULT_SOLVER_PARAMS: dict[str, dict] = {
    "lasso": {
        "alpha": 0.005,
        "tol": 1e-6,
        "fit_intercept": True,
    },
    "ridge": {
        "alpha": 1e-4,
        "tol": 1e-6,
        "fit_intercept": True,
        "max_iter": None,
        "solver": "auto",
        "positive": False,
        "random_state": None,
    },
    "lasso_lars": {
        "alpha": 1e-4,
        "fit_intercept": True,
        "max_iter": 500,
        "precompute": "auto",
        "eps": 1e-8,
    },
    "lasso_lars_cv": {
        "fit_intercept": True,
        "verbose": False,
        "max_iter": 500,
        "precompute": "auto",
        "max_n_alphas": 1000,
        "eps": 0.01,
    },
    "e-net": {
        "alpha": 1e-4,
        "tol": 1e-6,
        "l1_ratio": 0.6,
        "fit_intercept": True,
        "max_iter": 10000,
        "positive": False,
        "selection": "cyclic",
    },
    "e-net-cv": {
        "l1_ratio": 1.0,
        "eps": 0.05,
        "n_alphas": 300,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-3,
        "cv": None,
        "selection": "cyclic",
    },
    "mt-e-net": {
        "alpha": 1e-4,
        "tol": 1e-6,
        "l1_ratio": 0.7,
        "max_iter": 10000,
        "fit_intercept": True,
    },
}

# Mapping of solver names to scikit-learn regressors
_REGRESSOR_FACTORY = {
    "lasso": linear_model.Lasso,
    "ridge": linear_model.Ridge,
    "lasso_lars": linear_model.LassoLars,
    "lasso_lars_cv": linear_model.LassoLarsCV,
    "e-net": linear_model.ElasticNet,
    "e-net-cv": linear_model.MultiTaskElasticNetCV,
    "mt-e-net": linear_model.MultiTaskElasticNet,
}


class Scipy_Solver(Optimizer):
    def __init__(
        self,
        x_train,
        time_array,
        x_dot,
        library_object,
        x_0,
        x_raw,
        solver="lasso",
        solver_parameters = {},
        target="trajectory",
        scale_theta=False,
        objectives=None,
    ):
        super().__init__(
            x_train=x_train,
            time_array=time_array,
            x_dot=x_dot,
            library_object=library_object,
            x_0=x_0,
            x_raw=x_raw,
            solver=solver,
            target=target,
            scale_theta=scale_theta,
            objectives=objectives,
        )
        params = dict(_DEFAULT_SOLVER_PARAMS[solver])
        params.update(solver_parameters)
        print('parameters for regression model:', params)
        self.regression_model = _REGRESSOR_FACTORY[solver](**params)

    def _compute_metrics(self, x_row, regression):
        sol, x_ref, regression_coeff, opt_vars = self.solver(x_row, regression)
        objective_funcs = []

        for obj_func in self.objectives:
            try:
                objective_funcs.append(obj_func(self, sol))
            except Exception:
                objective_funcs.append(1e5)

        return objective_funcs, sol, regression_coeff, opt_vars

    def solver(self, optimization_vars, regression):
        opt_vars = optimization_vars[0] if len(optimization_vars) == 1 else optimization_vars
        self.alphas.append(opt_vars)

        self.library_object.set_base_funcs((opt_vars[0], opt_vars[1]))
        self.library_object.build(self.dimension)
        theta = self.library_object.evaluate(np.vstack(self.x_train))

        if self.scale_theta:
            scaler = StandardScaler()
            theta_scaled = scaler.fit_transform(theta)
            regression.fit(theta_scaled, self.x_dot)
            coeff_matrix = regression.coef_.T
            coeff_matrix_scaled = coeff_matrix / scaler.scale_[:, None]
            regression_coeff = coeff_matrix_scaled
        else:
            regression.fit(theta, self.x_dot)
            regression_coeff = regression.coef_.T

        sol_list = [
            odeint(self.library_object.get_rhs, x0, self.time_array, args=([opt_vars, regression_coeff],))
            for x0 in self.initial_values
        ]
        sol = np.vstack(sol_list)

        if self.target == "trajectory":
            x = sol
            x_ref = self.x_raw
        elif self.target == "derivative":
            x = theta @ regression_coeff
            x_ref = self.x_dot
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