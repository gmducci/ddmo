from typing import Any

import numpy as np
import pysindy as ps
from pysindy.optimizers import SINDyOptimizer
from scipy.integrate import solve_ivp

from ddmo.solver.base import BaseSolver
from ddmo.solver.build_feature_matrix import FeatureMatrixBuilder
from ddmo.solver.result import SolverResult
from ddmo.utils.array_helpers import (
    normalize_opt_vars,
    stack_state_matrix,
)
from ddmo.utils.data_constructor import DataConstructor


# Optimizer dispatch logic
def get_optimizer(optimizer_name: str):
    """
    Get optimizer class by name.
    """

    if optimizer_name == "STLSQ":
        from pysindy.optimizers import STLSQ

        return STLSQ
    elif optimizer_name == "SR3":
        from pysindy.optimizers import SR3

        return SR3
    elif optimizer_name == "ConstrainedSR3":
        from pysindy.optimizers import ConstrainedSR3

        return ConstrainedSR3
    elif optimizer_name == "FROLS":
        from pysindy.optimizers import FROLS

        return FROLS
    elif optimizer_name == "Lasso":
        from sklearn.linear_model import Lasso

        return Lasso
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


class Pysindy_Solver(BaseSolver):
    def __init__(
        self, data_instance: DataConstructor, optimizer: str, solver_parameters: dict,
        scale_features: bool = True
    ):
        self.data_instance = data_instance
        self.solver_parameters = solver_parameters
        self.optimizer = optimizer
        self.scale_features = scale_features
        self.builder = FeatureMatrixBuilder(data_instance)

    def configure(self, **kwargs):
        self.solver_parameters.update(kwargs)

    def solve(self, *args):
        theta = None
        lib_custom = None

        if len(args) == 1:
            # Build problem from optimization variables
            problem = self.builder.build_problem(args[0])
            lhs_matrix = problem.rhs
            theta = problem.feature_matrix
            lib_custom = problem.feature_library
            opt_vars = problem.opt_vars
            mask = None  # Mask is derived from optimizer result, not input here
        elif len(args) == 2:
            # Manual payload injection
            lhs_payload, rhs_payload = args
            lhs_matrix = stack_state_matrix(lhs_payload)
            if isinstance(rhs_payload, dict):
                opt_vars = normalize_opt_vars(rhs_payload.get("opt_vars"))
                theta = rhs_payload.get("theta")
                lib_custom = rhs_payload.get("library")
                mask = rhs_payload.get("colmask")
            else:
                opt_vars = normalize_opt_vars(rhs_payload)
        else:
            raise ValueError(
                "Pysindy_Solver.solve expects optimization variables or (lhs, rhs)."
            )

        if opt_vars is None:
            raise ValueError("Optimization variables are required for Pysindy solver.")

        theta = np.asarray(theta)

        from pysindy.utils import AxesArray
        from sklearn.preprocessing import StandardScaler

        params_dict = self.solver_parameters
        base_optimizer = get_optimizer(self.optimizer)(**params_dict)
        unbias_flag = getattr(base_optimizer, "unbias", True)
        sindy_optimizer = SINDyOptimizer(base_optimizer, unbias=unbias_flag)

        theta_axes = AxesArray(theta, {"ax_sample": 0, "ax_coord": 1})
        lhs_axes = AxesArray(np.asarray(lhs_matrix), {"ax_sample": 0, "ax_coord": 1})

        # Conditionally apply scaling
        if self.scale_features:
            # --- CHANGE START ---
            # Calculate L2 norm for Theta columns (library features)
            # Add epsilon to avoid division by zero for empty columns
            sx = np.linalg.norm(theta_axes, axis=0)
            sx[sx == 0] = 1.0 
            
            # Calculate std or norm for LHS (derivatives)
            # Std is usually fine for LHS as derivatives often center around 0
            sy = np.std(lhs_axes, axis=0) 
            sy[sy == 0] = 1.0

            # Apply scaling
            theta_axes = theta_axes / sx
            lhs_axes   = lhs_axes / sy
            # --- CHANGE END ---
        else:
            sx = np.ones(theta.shape[1])
            sy = np.ones(lhs_matrix.shape[1] if lhs_matrix.ndim > 1 else 1)

        # Fit optimizer
        try:
            fitted_optimizer = sindy_optimizer.fit(theta_axes, lhs_axes)
            raw_coefficients = fitted_optimizer.coef_
            
            # Unscale (Your logic here remains exactly valid)
            regression_coeff = (raw_coefficients * sy[:, None]) / sx[None, :]
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
                active_mask=mask,
            )

            sol_list = []

            for i in range(len(self.data_instance.x_0)):
                sol_ = model.simulate(
                    self.data_instance.x_0[i], self.data_instance.time_array
                )
                sol_list.append(sol_)
            sol = np.vstack(sol_list)

            x = sol

            return SolverResult(
                solution=sol,
                coefficients=regression_coeff,
                opt_vars=opt_vars,
                model=model,
                feature_matrix=theta,
            )
        except Exception as e:
            print(f"Error during solver execution: {e}")
            return SolverResult(
                solution=None,
                coefficients=None,
                opt_vars=opt_vars,
                model=None,
                feature_matrix=theta if theta is not None else None,
            )


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
            assert n_active == self._coefficients.shape[0], (
                "active_mask must match coefficients shape."
            )

    def coefficients(self) -> np.ndarray:
        return self._coefficients

    def simulate(
        self, x0: np.ndarray, time_array: np.ndarray, timeout: int = 10
    ) -> np.ndarray:
        import signal

        t_eval = np.asarray(time_array, dtype=float)

        # Divergence detection event - stops integration if values explode
        def divergence_event(t, y):
            MAX_VALUE = 1e10
            return MAX_VALUE - np.max(np.abs(y))

        divergence_event.terminal = True
        divergence_event.direction = -1

        # Timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Simulation timed out after {timeout} seconds")

        # Set up timeout (Unix/Mac only)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        except (AttributeError, ValueError):
            # signal.SIGALRM not available on Windows
            pass

        try:
            solution = solve_ivp(
                self._rhs,
                (t_eval[0], t_eval[-1]),
                np.asarray(x0, dtype=float),
                t_eval=t_eval,
                events=divergence_event,
                **self._solver_kwargs,
            )
        finally:
            # Cancel timeout
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass

        if not solution.success or len(solution.t) < len(t_eval):
            # Failed or terminated early due to divergence
            raise RuntimeError("Simulation failed or diverged")
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
