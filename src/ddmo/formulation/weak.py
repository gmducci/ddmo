from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from collections.abc import Sequence

import numpy as np
import pysindy as ps

from ddmo.formulation.base import Formulation
from ddmo.test_function_factory.base import TestFunctionFamily  # abstract base

IntegralSolver = Callable[..., np.ndarray]
Array = np.ndarray


@dataclass
class WeakFormulation(Formulation):
    """
    Weak formulation using a test-function family.

    This class does not own window_length or stride; those belong to the
    concrete TestFunctionFamily (e.g. HannWindow). It only queries the family
    for:
      • φ (prototype window, scaled)
      • dφ/dt (its time derivative, scaled)
      • centres on the time grid

    The integral computation in get_lhs/get_rhs is kept exactly as in the
    original implementation (slice + sum).
    """

    time_array: Array
    x_train: Array
    test_family: TestFunctionFamily
    integral_solver: IntegralSolver | None = None

    def __post_init__(self) -> None:
        # Basic grid info
        if self.time_array.ndim != 1:
            raise ValueError("time_array must be 1-D.")
        if self.x_train.ndim != 2:
            raise ValueError("x_train must be 2-D (M_time, n_var).")

        self.dt = float(self.time_array[1] - self.time_array[0])
        self.N = self.x_train.shape[0]          # number of time points

        # Ask the test family for its prototype window properties.
        # We assume the family has been instantiated with the right dt, window_length, stride, etc.
        try:
            self.window_length = int(self.test_family.window_length)   # e.g. 601
        except AttributeError as exc:
            raise AttributeError(
                "test_family must provide a `window_length` attribute."
            ) from exc

        # prototype window (φ) and its time derivative (dφ/dt),
        # already scaled according to the family's alpha/normalisation
        try:
            self.phi = np.asarray(self.test_family.phi, float)
            self.dphi = np.asarray(self.test_family.dphi, float)
        except AttributeError as exc:
            raise AttributeError(
                "test_family must provide `phi` and `dphi` attributes "
                "(prototype window and its derivative)."
            ) from exc

        if self.phi.shape[0] != self.window_length or self.dphi.shape[0] != self.window_length:
            raise ValueError(
                "test_family.phi and test_family.dphi must have length "
                f"window_length={self.window_length}."
            )

        # half-width for slicing
        self.half = self.window_length // 2

        # centres on the time grid: ask the family to compute them
        try:
            # this should create/refresh centres for the given grid length
            # you can name this method `grid_centers` or similar in HannWindow
            centers = self.test_family.grid_centers(self.N)
        except AttributeError as exc:
            raise AttributeError(
                "test_family must implement `grid_centers(M: int) -> np.ndarray` "
                "to provide window centres."
            ) from exc

        self.centers = np.asarray(centers, dtype=int)
        self.M = self.centers.shape[0]          # number of test windows

    # ------------------------------------------------------------------
    # Left-hand side (unchanged integral logic)
    # ------------------------------------------------------------------
    def get_lhs(self, *args: Any, **kwargs: Any) -> np.ndarray:

        targets = {}
        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)
        var_numbers = data_instance.x_train.shape[-1]
        B = np.zeros((self.M, var_numbers))

        for i, vn in enumerate(range(var_numbers)):
            b = np.zeros(self.M)
            xi = data_instance.x_train[:, i]
            for m, c in enumerate(self.centers):
                slice_range = slice(c - self.half, c + self.half + 1)
                B[m, i] = -np.sum(xi[slice_range] * self.dphi) * self.dt
            # targets[f"var_{vn}"] = b

        return B

    # ------------------------------------------------------------------
    # Right-hand side (unchanged integral logic)
    # ------------------------------------------------------------------
    def get_rhs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:

        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)

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