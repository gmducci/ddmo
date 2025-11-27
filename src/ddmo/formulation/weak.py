from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
from collections.abc import Sequence

import numpy as np
import pysindy as ps

from ddmo.formulation.base import Formulation
from ddmo.test_function_factory.base import TestFunctionFamily  # abstract base
from ddmo.test_function_factory.registry import WEAK_TEST_FAMILY_REGISTRY

IntegralSolver = Callable[..., np.ndarray]
FamilyBuilder = Callable[[dict[str, float]], TestFunctionFamily]  # <- dict, not tuple

Array = np.ndarray


@dataclass
class WeakFormulation(Formulation):
    """
    Weak formulation driven by a test-function family chosen from a registry.

    Parameters
    ----------
    time_array : (M,)
        Time grid.
    x_train : (M, n_var)
        Training state trajectory.
    class_attributes : dict
        Dictionary that must at least contain:
            - "test_functions_class": name in WEAK_TEST_FAMILY_REGISTRY
        and may contain any other keyword arguments for that test-family
        (e.g. "window_length", "stride", ...).
    """
    time_array: Array
    x_train: Array
    class_attributes: dict[str, Any]
    integral_solver: IntegralSolver | None = None
    family_builder: Optional[FamilyBuilder] = None

    def __post_init__(self) -> None:
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
            for m, c in enumerate(self.centers):
                slice_range = slice(c - self.half, c + self.half + 1)
                B[m, i] = -np.sum(xi[slice_range] * self.dphi) * self.dt

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

        named_funcs = data_instance.library_object([*opt_vars])
        functions = [func for func, _ in named_funcs]
        raw_names = [name for _, name in named_funcs]
        function_names = [_ensure_callable_name(name) for name in raw_names]

        x_matrix = _stack_state_matrix(data_instance.x_train)

        lib_custom = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names,
        )
        lib_custom.fit(x_matrix)
        theta = lib_custom.transform(x_matrix)
        colmask = ~np.isnan(theta).any(axis=0)

        theta = theta[:, colmask]
        rhs = np.zeros((self.M, theta.shape[1]))
        for m, c in enumerate(self.centers):
            slice_range = slice(
                c - self.window_length // 2,
                c + self.window_length // 2 + 1,
            )
            for j, G in enumerate(theta.T):
                rhs[m, j] = np.sum(G[slice_range] * self.phi) * self.dt

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
        msg = "StrongFormulation.get_rhs requires a data_instance."
        raise ValueError(msg)

    if opt_vars is None:
        msg = "StrongFormulation.get_rhs requires optimization variables."
        raise ValueError(msg)

    normalized_opt_vars = _normalize_opt_vars(opt_vars)
    return data_instance, normalized_opt_vars


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


def _ensure_callable_name(name: Any) -> Callable[..., str]:
    if callable(name):
        return name

    label = str(name)
    return lambda *_, label=label: label
