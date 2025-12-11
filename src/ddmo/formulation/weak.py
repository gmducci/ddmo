from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
from collections.abc import Sequence

import numpy as np
import pysindy as ps

from ddmo.utils.data_constructor import DataConstructor
from ddmo.formulation.base import Formulation
from ddmo.test_function_factory.base import TestFunctionFamily  # abstract base
from ddmo.test_function_factory.registry import WEAK_TEST_FAMILY_REGISTRY
from ddmo.utils.array_helpers import normalize_opt_vars, stack_state_matrix, ensure_callable_name

IntegralSolver = Callable[..., np.ndarray]
FamilyBuilder = Callable[[dict[str, float]], TestFunctionFamily]  # <- dict, not tuple

Array = np.ndarray


@dataclass
class WeakFormulation(Formulation):
    """
    Weak formulation driven by a test-function family chosen from a registry.

    Parameters
    ----------
    data_instance: DataConstructor
        Object containing time_array and x_train.
    class_attributes : dict
        Dictionary that must at least contain:
            - "test_functions_class": name in WEAK_TEST_FAMILY_REGISTRY
        and may contain any other keyword arguments for that test-family
        (e.g. "window_length", "stride", ...).
    """
    data_instance: DataConstructor
    class_attributes: dict[str, Any]
    integral_solver: IntegralSolver | None = None
    family_builder: Optional[FamilyBuilder] = None

    def __post_init__(self) -> None:
        self.time_array = self.data_instance.time_array
        self.x_train = self.data_instance.x_train
        
        # Basic shape checks
        if self.time_array.ndim != 1:
            raise ValueError("time_array must be 1-D.")
        if self.x_train.ndim != 2:
            raise ValueError("x_train must be 2-D (M_time, n_var).")

        self.dt = float(self.time_array[1] - self.time_array[0])
        self.N = self.x_train.shape[0]

        # ------------------------------------------------------------------
        # Build initial test-family from the registry
        # ------------------------------------------------------------------
        family_name = self.class_attributes.get("test_functions_class")
        if family_name is None:
            raise ValueError(
                "class_attributes must contain key 'test_functions_class'."
            )

        try:
            family_cls = WEAK_TEST_FAMILY_REGISTRY[family_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown test-function family '{family_name}'. "
                f"Available: {sorted(WEAK_TEST_FAMILY_REGISTRY)}"
            ) from exc

        # All other entries in class_attributes are treated as parameters
        base_params = {
            k: v for k, v in self.class_attributes.items()
            if k != "test_functions_class"
        }

        # Build the initial family and configure the formulation
        initial_family = family_cls(dt=self.dt, **base_params)
        self._configure_from_family(initial_family)

        # ------------------------------------------------------------------
        # Build a family_builder closure for later updates (GA, etc.)
        # ------------------------------------------------------------------
        def builder(param_dict: dict[str, float]) -> TestFunctionFamily:
            """
            Create a new test-family instance from updated optimisation parameters.

            param_dict maps parameter names (e.g. 'gamma_1', 'stride') to floats.
            """
            params = dict(base_params)

            for name, val in param_dict.items():
                if name not in params:
                    # ignore optimisation variables that do not correspond
                    # to test-family parameters
                    continue

                if name == "stride":
                    params[name] = int(round(val))
                elif name == "window_length":
                    params[name] = int(round(val)) * 2 + 1
                else:
                    params[name] = float(val)

            return family_cls(dt=self.dt, **params)

        self.family_builder = builder

    # ------------------------------------------------------------------
    # Configuration from a TestFunctionFamily (unchanged)
    # ------------------------------------------------------------------
    def _configure_from_family(self, family: TestFunctionFamily) -> None:
        self.test_family = family

        try:
            phi = np.asarray(family.phi, float)
            dphi = np.asarray(family.dphi, float)
        except AttributeError as exc:
            raise AttributeError(
                "test_family must provide `phi` and `dphi` attributes "
                "(prototype test function and its derivative)."
            ) from exc

        if phi.shape != dphi.shape or phi.ndim != 1:
            raise ValueError(
                "test_family.phi and test_family.dphi must be 1-D arrays "
                "of the same length."
            )

        window_length = phi.shape[0]

        self.window_length = window_length
        self.phi = phi
        self.dphi = dphi
        self.half = window_length // 2
        # 1) quadrature weights (currently: trapezoidal rule)
        q = np.full(window_length, self.dt)  # q_k = dt for all k
        q[0] = q[-1] = 0.5 * self.dt   # end points have half weight


        # precompute weights for this family (current rectangle rule)
        # 2) combine quadrature with test function
        refinement = getattr(family, "refinement", 16)

        self.w_rhs = _build_rhs_weights_with_interpolation(
            phi=self.phi,
            dt=self.dt,
            refinement=refinement,
        )
        self.w_lhs = _build_lhs_weights_with_interpolation(
            dphi=self.dphi,
            dt=self.dt,
            refinement=refinement,
        )

        try:
            centers = family.grid_centers(self.N)
        except AttributeError as exc:
            raise AttributeError(
                "test_family must implement `grid_centers(M: int) -> np.ndarray` "
                "to provide window centres."
            ) from exc

        self.centers = np.asarray(centers, dtype=int)
        self.M = self.centers.shape[0]

    # ------------------------------------------------------------------
    # Left-hand side (unchanged integral logic)
    # ------------------------------------------------------------------
    def get_lhs(self, *args: Any, **kwargs: Any) -> np.ndarray:
        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)

        param_dict = kwargs.get("opt_param_dict", None)
        if self.family_builder is not None and param_dict is not None:
            new_family = self.family_builder(param_dict)
            self._configure_from_family(new_family)

        var_numbers = data_instance.x_train.shape[-1]
        B = np.zeros((self.M, var_numbers))

        for i in range(var_numbers):
            xi = data_instance.x_train[:, i]
            B[:, i] = _sliding_integral(xi, self.centers, self.w_lhs)

        return B
    # ------------------------------------------------------------------
    # Right-hand side (unchanged integral logic)
    # ------------------------------------------------------------------
    def get_rhs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)

        param_dict = kwargs.get("opt_param_dict", None)
        if self.family_builder is not None and param_dict is not None:
            new_family = self.family_builder(param_dict)
            self._configure_from_family(new_family)

        # Pass dict to library function (safer than positional list)
        # If param_dict not available, fall back to list for backward compat
        if param_dict is not None:
            named_funcs = data_instance.library_object(param_dict)
        else:
            named_funcs = data_instance.library_object([*opt_vars])
        functions = [func for func, _ in named_funcs]
        raw_names = [name for _, name in named_funcs]
        function_names = [ensure_callable_name(name) for name in raw_names]

        x_matrix = stack_state_matrix(data_instance.x_train)

        lib_custom = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names,
        )
        lib_custom.fit(x_matrix)
        theta = lib_custom.transform(x_matrix)
        colmask = ~np.isnan(theta).any(axis=0)

        theta = theta[:, colmask]
        rhs = np.zeros((self.M, theta.shape[1]))

        for j, G in enumerate(theta.T):
            rhs[:, j] = _sliding_integral(G, self.centers, self.w_rhs)

        return {
            "theta": rhs,
            "library": lib_custom,
            "opt_vars": opt_vars,
            "named_funcs": named_funcs,
            "colmask": colmask,
        }
    
# ----------------------------------------------------------------------
# Helpers (unchanged)
# ----------------------------------------------------------------------
def _extract_rhs_inputs(*args: Any, **kwargs: Any) -> tuple[Any, tuple[float, ...]]:
    data_instance = None
    opt_vars = None

    if args:
        data_instance = args[0]
        if len(args) > 1:
            opt_vars = args[1]

    data_instance = kwargs.get("data_instance", data_instance)
    opt_vars = kwargs.get("optimization_variables", opt_vars)

    if data_instance is None:
        msg = "WeakFormulation.get_rhs requires a data_instance."
        raise ValueError(msg)

    if opt_vars is None:
        msg = "WeakFormulation.get_rhs requires optimization variables."
        raise ValueError(msg)

    normalized_opt_vars = normalize_opt_vars(opt_vars)
    return data_instance, normalized_opt_vars


def _sliding_integral(
    signal: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute windowed integrals using precomputed weights.

    signal  : shape (N,)
    centers : shape (M,), integer indices of window centres
    weights : shape (L,), precomputed weights on a reference window

    Returns
    -------
    out : shape (M,)
        out[m] = sum_{k=0}^{L-1} signal[c_m - h + k] * weights[k]
    """
    L = weights.shape[0]
    h = L // 2
    out = np.empty(centers.shape[0], dtype=float)

    for m, c in enumerate(centers):
        sl = slice(c - h, c + h + 1)
        out[m] = np.dot(signal[sl], weights)

    return out

def _build_lhs_weights_with_interpolation(
    dphi: np.ndarray,
    dt: float,
    refinement: int = 4,
) -> np.ndarray:
    """
    Build left-hand-side weak weights w_lhs for one window, using
    piecewise-linear interpolation in time and a refined quadrature.

    Parameters
    ----------
    dphi : (L,)
        Discrete samples of φ'(t) on the window (prototype test function derivative).
    dt : float
        Coarse time step between consecutive samples t_n.
    refinement : int, optional
        Number of fine evaluation points per coarse interval [t_n, t_{n+1}].
        Must be >= 2. refinement=2 reduces to a coarse trapezoid-like rule.

    Returns
    -------
    w_lhs : (L,)
        Weights such that, for a windowed signal x[n],
            ∫ x_t(t) φ(t) dt  ≈  sum_k x_k * w_lhs[k]
        (over that window).
    """
    dphi = np.asarray(dphi, dtype=float)
    L = dphi.shape[0]

    if L < 2:
        raise ValueError("Window length must be at least 2 to build LHS weights.")

    if refinement < 2:
        raise ValueError("refinement must be >= 2.")

    w = np.zeros(L, dtype=float)

    # Loop over coarse intervals [n, n+1] within the window
    for n in range(L - 1):
        dphi_n = dphi[n]
        dphi_np1 = dphi[n + 1]

        # Fine grid over local coordinate s ∈ [0, 1]
        s = np.linspace(0.0, 1.0, refinement)

        # Piecewise-linear interpolation of φ' between dphi_n and dphi_np1
        dphi_fine = (1.0 - s) * dphi_n + s * dphi_np1

        # Assume x(t) is linear between x_n and x_{n+1}:
        #   x(t(s)) = x_n(1 - s) + x_{n+1} s
        #
        # Then for the weak derivative:
        #   ∫ x_t φ dt = -∫ x φ' dt
        #
        # Contribution of this interval to the integral -∫ x φ' dt can be written as
        #   I_n ≈ A_n x_n + B_n x_{n+1}
        #
        # with
        #   A_n = -∫ (1 - s) φ'(t(s)) dt
        #   B_n = -∫ s       φ'(t(s)) dt
        #
        # Change of variable t = t_n + s dt ⇒ dt = dt * ds,
        # so we just integrate over s with dx = dt / (refinement - 1).

        integrand_left = (1.0 - s) * dphi_fine
        integrand_right = s * dphi_fine

        # Numerical quadrature on the fine grid (trapezoidal)
        dx_fine = dt / (refinement - 1)

        A_n = -np.trapz(integrand_left, dx=dx_fine)   # coefficient for x_n
        B_n = -np.trapz(integrand_right, dx=dx_fine)  # coefficient for x_{n+1}

        # Accumulate these interval contributions into nodal weights
        w[n] += A_n
        w[n + 1] += B_n

    return w

def _build_rhs_weights_with_interpolation(
    phi: np.ndarray,
    dt: float,
    refinement: int = 4,
) -> np.ndarray:
    """
    Build right-hand-side weak weights w_rhs for one window, using
    piecewise-linear interpolation in time and a refined quadrature.

    Parameters
    ----------
    phi : (L,)
        Discrete samples of φ(t) on the window (prototype test function).
    dt : float
        Coarse time step between consecutive samples t_n.
    refinement : int, optional
        Number of fine evaluation points per coarse interval [t_n, t_{n+1}].
        Must be >= 2.

    Returns
    -------
    w_rhs : (L,)
        Weights such that, for a windowed signal x[n],
            ∫ x(t) φ(t) dt  ≈  sum_k x_k * w_rhs[k]
        (over that window).
    """
    phi = np.asarray(phi, dtype=float)
    L = phi.shape[0]

    if L < 2:
        raise ValueError("Window length must be at least 2 to build RHS weights.")

    if refinement < 2:
        raise ValueError("refinement must be >= 2.")

    w = np.zeros(L, dtype=float)

    # Loop over coarse intervals [n, n+1] within the window
    for n in range(L - 1):
        phi_n = phi[n]
        phi_np1 = phi[n + 1]

        # Fine grid over s ∈ [0, 1]
        s = np.linspace(0.0, 1.0, refinement)

        # Piecewise-linear interpolation of φ
        phi_fine = (1.0 - s) * phi_n + s * phi_np1

        # Assume x(t) is linear between x_n and x_{n+1}:
        #   x(t(s)) = x_n(1 - s) + x_{n+1} s
        #
        # Then
        #   ∫ x φ dt ≈ A_n x_n + B_n x_{n+1}
        # with
        #   A_n = ∫ (1 - s) φ(t(s)) dt
        #   B_n = ∫ s       φ(t(s)) dt
        #
        # t = t_n + s dt ⇒ dt = dt * ds

        integrand_left = (1.0 - s) * phi_fine
        integrand_right = s * phi_fine

        dx_fine = dt / (refinement - 1)

        A_n = np.trapz(integrand_left, dx=dx_fine)
        B_n = np.trapz(integrand_right, dx=dx_fine)

        w[n] += A_n
        w[n + 1] += B_n

    return w