import numpy as np
import pysindy as ps
from collections.abc import Sequence
from typing import Any
from pysindy.optimizers import (
    SR3,
    STLSQ,
    ConstrainedSR3,
    EnsembleOptimizer,
    SINDyOptimizer,
)
from sklearn.linear_model import Lasso
from ddmo.utils.data_constructor import DataConstructor
from ddmo.solver.base import BaseSolver
from scipy.integrate import solve_ivp
from pysindy.utils import AxesArray
from sklearn.preprocessing import StandardScaler



class Pysindy_Solver(BaseSolver):
    def __init__(self, 
                 data_instance: DataConstructor,
                 optimizer: str,
                 solver_parameters: dict):
        
        self.data_instance = data_instance
        self.solver_parameters = solver_parameters
        self.optimizer = optimizer

    def configure(self, **kwargs):
        self.solver_parameters.update(kwargs)

    def solve(self, *args):

        theta = None
        lib_custom = None
        named_funcs = None

        lhs_matrix = self._stack_state_matrix(self.data_instance.x_dot)

        if len(args) == 1:
            optimization_vars = args[0]
        elif len(args) == 2:
            lhs_payload, rhs_payload = args
            lhs_matrix = self._stack_state_matrix(lhs_payload)
            if isinstance(rhs_payload, dict):
                optimization_vars = rhs_payload.get("opt_vars")
                theta = rhs_payload.get("theta")
                lib_custom = rhs_payload.get("library")
                named_funcs = rhs_payload.get("named_funcs")
                mask = rhs_payload.get("colmask")
            else:
                optimization_vars = rhs_payload
        else:
            raise ValueError("Pysindy_Solver.solve expects optimization variables or (lhs, rhs).")

        if optimization_vars is None:
            raise ValueError("Optimization variables are required for Pysindy solver.")

        opt_vars = self._normalize_opt_vars(optimization_vars)

        # Optimizer dispatch logic
        _optimizers = {
            "STLSQ": STLSQ,
            "SR3": SR3,
            "ConstrainedSR3": ConstrainedSR3,
            "EnsembleOptimizer": EnsembleOptimizer,
            "Lasso": Lasso}

        x_matrix = self._stack_state_matrix(self.data_instance.x_train)

        theta = np.asarray(theta)

        params_dict = self.solver_parameters
        base_optimizer = _optimizers[self.optimizer](**params_dict)
        unbias_flag = getattr(base_optimizer, "unbias", True)
        sindy_optimizer = SINDyOptimizer(base_optimizer, unbias=unbias_flag)

        theta_axes = AxesArray(theta, {"ax_sample": 0, "ax_coord": 1})
        lhs_axes = AxesArray(np.asarray(lhs_matrix), {"ax_sample": 0, "ax_coord": 1})
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        theta_axes = scaler_X.fit_transform(theta_axes)
        lhs_axes = scaler_y.fit_transform(lhs_axes)
        sx = np.asarray(scaler_X.scale_)   # length P
        sy = np.asarray(scaler_y.scale_)   # length d

        # Increase threshold to enforce more sparsity
        try:
            fitted_optimizer = sindy_optimizer.fit(theta_axes, lhs_axes)
            raw_coefficients = fitted_optimizer.coef_
            # regression_coeff = raw_coefficients.T
            regression_coeff   = (raw_coefficients * sy[:, None]) / sx[None, :]
            regression_coeff = regression_coeff.T

            solver_kwargs = {
                "method": "BDF",
                "rtol": 1e-7,
                "atol": 1e-9,
            }
            model = _ManualSINDyModel(
                feature_library=lib_custom,
                coefficients=regression_coeff,
                optimizer=fitted_optimizer.optimizer,
                solver_kwargs=solver_kwargs,
                active_mask=mask
            )

            sol_list = []

            for i in range(len(self.data_instance.x_0)):
                sol_ = model.simulate(self.data_instance.x_0[i], self.data_instance.time_array)
                sol_list.append(sol_)
            sol = np.vstack(sol_list)

            x = sol

            return x, regression_coeff, opt_vars, model, theta
        except Exception as e:
            print(f"Error during solver execution: {e}")
            return None, None, None, None, None

    @staticmethod
    def _normalize_opt_vars(opt_vars: Any) -> tuple[float, ...]:
        if np.isscalar(opt_vars):
            return (float(opt_vars),)

        if isinstance(opt_vars, np.ndarray):
            flattened = opt_vars.astype(float).ravel()
            return tuple(float(value) for value in flattened)

        if isinstance(opt_vars, Sequence) and not isinstance(opt_vars, (str, bytes)):
            return tuple(float(value) for value in opt_vars)

        try:
            return (float(opt_vars),)
        except (TypeError, ValueError) as exc:
            msg = "Cannot normalize optimization variables into a tuple of floats."
            raise TypeError(msg) from exc

    @staticmethod
    def _stack_state_matrix(values: Any) -> np.ndarray:
        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                return values.reshape(-1, 1)
            if values.ndim == 3:
                n_samples = values.shape[0] * values.shape[1]
                return values.reshape(n_samples, values.shape[-1])
            return values

        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            stacked = [np.atleast_2d(np.asarray(block)) for block in values]
            return np.vstack(stacked)

        return np.atleast_2d(np.asarray(values))

    @staticmethod
    def _ensure_callable_name(name: Any):
        if callable(name):
            return name

        label = str(name)
        return lambda *_, label=label: label
    
    @staticmethod
    def _get_rhs(t, ssp, matrices):
        # This implementation needs to be revised for non-autonomous systems!!!
        lib_custom = matrices[0]
        coeff_matrix = matrices[1]
        # No need to recompute theta!!! Take raw-wise elements from build_theta output!!!
        theta = lib_custom.transform(np.atleast_2d(ssp))
        rhs = theta @ coeff_matrix

        return rhs.ravel()


class _ManualSINDyModel:
    def __init__(
        self,
        feature_library: ps.CustomLibrary,
        coefficients: np.ndarray,
        optimizer: Any,
        solver_kwargs: dict[str, Any] | None = None,
        active_mask: np.ndarray | None = None,
    ) -> None:
        """
        coefficients: shape (n_active_features, n_states)
        active_mask: boolean array of shape (n_all_features,) indicating
                     which columns of the full library output were kept.
        """
        self.feature_library = feature_library
        self._coefficients = coefficients
        self.optimizer = optimizer
        self._solver_kwargs = dict(solver_kwargs or {})
        self.active_mask = active_mask

        if self.active_mask is not None:
            n_active = np.count_nonzero(self.active_mask)
            assert (
                n_active == self._coefficients.shape[0]
            ), "active_mask must match coefficients shape."

    def coefficients(self) -> np.ndarray:
        return self._coefficients

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
            raise RuntimeError("Manual SINDy simulation failed")
        return solution.y.T

    def _rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        # shape: (1, n_features_full)
        theta_full = self.feature_library.transform(np.atleast_2d(state))

        if self.active_mask is not None:
            # same trimming as during training
            theta = theta_full[:, self.active_mask]
        else:
            theta = theta_full

        # coefficients: (n_active_features, n_states)
        return (theta @ self._coefficients).ravel()