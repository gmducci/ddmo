from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
from collections.abc import Sequence

import numpy as np
import pysindy as ps

from ddmo.formulation.base import Formulation
from ddmo.test_function_factory.base import TestFunctionFamily  # abstract base

IntegralSolver = Callable[..., np.ndarray]
FamilyBuilder = Callable[[tuple[float, ...]], TestFunctionFamily]

Array = np.ndarray


@dataclass
class WeakFormulation(Formulation):
    time_array: Array
    x_train: Array
    test_family: TestFunctionFamily
    integral_solver: IntegralSolver | None = None
    family_builder: Optional[FamilyBuilder] = None

    def __post_init__(self) -> None:
        if self.time_array.ndim != 1:
            raise ValueError("time_array must be 1-D.")
        if self.x_train.ndim != 2:
            raise ValueError("x_train must be 2-D (M_time, n_var).")

        self.dt = float(self.time_array[1] - self.time_array[0])
        self.N = self.x_train.shape[0]

        # new helper
        self._configure_from_family(self.test_family)

    def _configure_from_family(self, family: TestFunctionFamily) -> None:
        """
        Read all necessary information from a TestFunctionFamily instance
        and cache it in this WeakFormulation.

        This is the only place where we touch family-specific attributes
        like `phi`, `dphi`, `grid_centers`. We do NOT rely on parameter
        names such as window_length, stride, sigma, etc.
        """
        self.test_family = family

        # 1) Get prototype window and its derivative (generic arrays)
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

        # 2) Derive window length from phi itself (no attribute name needed)
        window_length = phi.shape[0]

        self.window_length = window_length
        self.phi = phi
        self.dphi = dphi
        self.half = window_length // 2

        # 3) Centres on the time grid: generic method from the base class
        try:
            centers = family.grid_centers(self.N)   # implemented by each subclass
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

        # NEW: if a builder is provided, rebuild the family for this candidate
        if self.family_builder is not None:
            new_family = self.family_builder(opt_vars)
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

        # NEW: same hook here
        if self.family_builder is not None:
            new_family = self.family_builder(opt_vars)
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