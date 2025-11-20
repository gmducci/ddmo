from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from collections.abc import Sequence
import pysindy as ps

import numpy as np

from .base import Formulation

IntegralSolver = Callable[..., np.ndarray]

def get_test_function_class(test_function_class: str):
    """
    Get test function class by name.

    Args:
        test_function_factory: Name of the optimizer to retrieve

    Returns:
        Optimizer class

    Raises:
        ValueError: If optimizer name is not recognized
    """

    if test_function_class.lower() == "hannwindow":
        from ddmo.test_function_factory.hann_window import  HannWindow
        return HannWindow
    else:
        raise ValueError(f"Unknown test function: {test_function_class}")



@dataclass
class WeakFormulation(Formulation):
    """
    Weak formulation helper that delegates integral evaluation to a solver.

    The default behaviour mirrors the sketch in the design notes: both sides of
    the equation use the same integral, but the left-hand side carries a minus
    sign.
    """
    
    def __init__(self, time_array: np.ndarray,
                 x_train: np.ndarray,
                 window_length: int,
                 stride: int,
                 test_function_class
                 ):
        self.window_length = window_length
        self.test_function_class = test_function_class
        self.dt = time_array[1] - time_array[0]
        self.N = len(x_train)
        self.test_function = test_function_class.evaluate(window_length)
        self.test_function_derivatives = test_function_class.evaluate_derivative(self.test_function, self.dt)
        #self_derivative_test_function = _window_derivative(self.test_function, self.dt)
        # normalization factor for the integral approximation
        self.alpha = 1 / np.sum(self.test_function) * self.dt

        self.phi = self.test_function * self.alpha
        self.dphi = self.test_function_derivatives * self.alpha
        self.half = window_length // 2
        self.centers = np.arange(self.half, self.N - self.half, stride)


        self.M = len(self.centers)
    
    integral_solver: IntegralSolver | None = None

    # def solve_integral(self, *args: Any, **kwargs: Any) -> np.ndarray:
    #     if self.integral_solver is None:
    #         msg = "An integral_solver callable must be provided for weak formulations."
    #         raise ValueError(msg)
        
    #     return self.integral_solver(*args, **kwargs)

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

    def get_rhs(self, *args: Any, **kwargs: Any) -> np.ndarray:
        
        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)

        named_funcs = data_instance.library_object([*opt_vars])
        functions = [func for func, _ in named_funcs]
        raw_names = [name for _, name in named_funcs]
        function_names = [_ensure_callable_name(name) for name in raw_names]

        x_matrix = _stack_state_matrix(data_instance.x_train)

        lib_custom = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names
        )
        lib_custom.fit(x_matrix)
        theta = lib_custom.transform(x_matrix)
        colmask = ~np.isnan(theta).any(axis=0)

        theta = theta[:, colmask] 
        rhs = np.zeros((self.M, theta.shape[1]))
        for m, c in enumerate(self.centers):
            slice_range = slice(c - self.window_length // 2, c + self.window_length // 2 + 1)
            for j, G in enumerate(theta.T):
                rhs[m, j] = np.sum(G[slice_range] * self.phi) * self.dt

        return {
            "theta": rhs,
            "library": lib_custom,
            "opt_vars": opt_vars,
            "named_funcs": named_funcs,
            "colmask": colmask,
        }
    
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
