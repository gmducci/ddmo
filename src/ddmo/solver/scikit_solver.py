from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import (
    ElasticNetCV,
    Lasso,
    LassoCV,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
    Ridge,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from ddmo.solver.base import BaseSolver
from ddmo.utils.array_helpers import normalize_opt_vars, stack_state_matrix
from ddmo.utils.data_constructor import DataConstructor


class ScikitSolver(BaseSolver):
    def __init__(
        self, data_instance: DataConstructor, optimizer: str, solver_parameters: dict
    ):
        self.data_instance = data_instance
        self.solver_parameters = solver_parameters or {}
        self.optimizer_name = optimizer

        # Validate optimizer
        valid_optimizers = [
            "LassoCV",
            "ElasticNetCV",
            "MultiTaskLassoCV",
            "MultiTaskElasticNetCV",
            "Lasso",
            "Ridge",
        ]
        if self.optimizer_name not in valid_optimizers:
            raise ValueError(f"ScikitSolver only supports: {valid_optimizers}")

    def configure(self, **kwargs):
        self.solver_parameters.update(kwargs)

    def solve(self, *args):
        # 1. Parse Arguments
        theta = None
        lib_custom = None
        mask = None

        lhs_matrix = stack_state_matrix(self.data_instance.x_dot)

        if len(args) == 1:
            optimization_vars = args[0]
        elif len(args) == 2:
            lhs_payload, rhs_payload = args
            lhs_matrix = stack_state_matrix(lhs_payload)
            if isinstance(rhs_payload, dict):
                optimization_vars = rhs_payload.get("opt_vars")
                theta = rhs_payload.get("theta")
                lib_custom = rhs_payload.get("library")
                mask = rhs_payload.get("colmask")
            else:
                optimization_vars = rhs_payload
        else:
            raise ValueError(
                "ScikitSolver.solve expects optimization variables or (lhs, rhs)."
            )

        if optimization_vars is None:
            raise ValueError("Optimization variables are required for ScikitSolver.")

        opt_vars = normalize_opt_vars(optimization_vars)

        # 2. Prepare Data
        if theta is None:
            pass

        theta = np.asarray(theta)

        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # Check dimensions
        if theta.ndim != 2:
            pass

        X_scaled = scaler_X.fit_transform(theta)
        y_scaled = scaler_y.fit_transform(lhs_matrix)

        sx = np.asarray(scaler_X.scale_)  # length n_features
        sy = np.asarray(scaler_y.scale_)  # length n_targets

        # 3. Instantiate Optimizer
        params = self.solver_parameters.copy()

        is_multi_output = y_scaled.shape[1] > 1
        model = None

        if self.optimizer_name == "LassoCV":
            base_model = LassoCV(**params)
            if is_multi_output:
                model = MultiOutputRegressor(base_model)
            else:
                model = base_model
        elif self.optimizer_name == "ElasticNetCV":
            base_model = ElasticNetCV(**params)
            if is_multi_output:
                model = MultiOutputRegressor(base_model)
            else:
                model = base_model
        elif self.optimizer_name == "MultiTaskLassoCV":
            model = MultiTaskLassoCV(**params)
        elif self.optimizer_name == "MultiTaskElasticNetCV":
            model = MultiTaskElasticNetCV(**params)
        elif self.optimizer_name == "Lasso":
            base_model = Lasso(**params)
            # Default to MultiOutputRegressor for Lasso to allow independent sparsity
            if is_multi_output:
                model = MultiOutputRegressor(base_model)
            else:
                model = base_model
        elif self.optimizer_name == "Ridge":
            # Ridge handles multi-output natively
            model = Ridge(**params)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer_name}")

        # 4. Fit
        try:
            # Handle fit_intercept default if not present
            if 'fit_intercept' not in params:
                # Check if the model has this attribute or accepts it in init
                # For MultiOutputRegressor, we set it on the estimator.
                # We already passed params to init, so if it wasn't there, it used default.
                # Sklearn linear models default fit_intercept=True. SINDy often wants False.
                # But let's respect user params or default to True if they didn't specify.
                # If we wanted to force False, we should have done it in params dict before init.
                pass

            model.fit(X_scaled, y_scaled)

            # 5. Extract Coefficients
            if isinstance(model, MultiOutputRegressor):
                # model.estimators_ is a list of fitted estimators
                # each has .coef_ of shape (n_features,)
                raw_coefficients = np.array([est.coef_ for est in model.estimators_])
                # shape (n_targets, n_features)
            else:
                # LassoCV/ElasticNetCV (single output) -> coef_ is (n_features,)
                # MultiTask -> coef_ is (n_targets, n_features)
                raw_coefficients = model.coef_
                if raw_coefficients.ndim == 1:
                    raw_coefficients = raw_coefficients.reshape(1, -1)

            # Rescale
            # coeff = raw * (sy / sx)
            # raw is (n_targets, n_features)
            # sy is (n_targets,) -> (n_targets, 1)
            # sx is (n_features,) -> (1, n_features)

            regression_coeff = (raw_coefficients * sy[:, None]) / sx[None, :]

            # Convert to (n_features, n_targets) as per SINDy convention
            regression_coeff = regression_coeff.T

            # 6. Simulation
            solver_kwargs = {
                "method": "BDF",
                "rtol": 1e-7,
                "atol": 1e-9,
            }

            sim_model = _ManualSimulationModel(
                feature_library=lib_custom,
                coefficients=regression_coeff,
                solver_kwargs=solver_kwargs,
                active_mask=mask,
            )

            sol_list = []
            if self.data_instance and self.data_instance.x_0 is not None:
                for i in range(len(self.data_instance.x_0)):
                    sol_ = sim_model.simulate(
                        self.data_instance.x_0[i], self.data_instance.time_array
                    )
                    sol_list.append(sol_)
                x_sim = np.vstack(sol_list)
            else:
                x_sim = None

            return x_sim, regression_coeff, opt_vars, sim_model, theta

        except Exception as e:
            print(f"Error during ScikitSolver execution: {e}")
            return None, None, None, None, None


class _ManualSimulationModel:
    def __init__(
        self,
        feature_library: Any,  # Expecting ps.CustomLibrary or similar with .transform
        coefficients: np.ndarray,
        solver_kwargs: dict[str, Any] | None = None,
        active_mask: np.ndarray | None = None,
    ) -> None:
        self.feature_library = feature_library
        self._coefficients = coefficients
        self._solver_kwargs = dict(solver_kwargs or {})
        self.active_mask = active_mask

        if self.active_mask is not None:
            n_active = np.count_nonzero(self.active_mask)
            # Check validation?
            # Pysindy Solver asserts this
            pass

    def simulate(self, x0: np.ndarray, time_array: np.ndarray) -> np.ndarray:
        t_eval = np.asarray(time_array, dtype=float)
        solution = solve_ivp(
            self._rhs,
            (t_eval[0], t_eval[-1]),
            np.asarray(x0, dtype=float),
            t_eval=t_eval,
            **self._solver_kwargs,
        )
        if not solution.success:
            raise RuntimeError("Manual Simulation failed")
        return solution.y.T

    def _rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        # shape: (1, n_features_full)
        if self.feature_library is None:
            raise ValueError("Feature library required for simulation")

        theta_full = self.feature_library.transform(np.atleast_2d(state))

        if self.active_mask is not None:
            theta = theta_full[:, self.active_mask]
        else:
            theta = theta_full

        # coefficients: (n_active_features, n_states)
        return (theta @ self._coefficients).ravel()
