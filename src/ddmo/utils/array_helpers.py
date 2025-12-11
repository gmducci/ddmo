"""
Shared array manipulation utilities used across formulations and solvers.
"""
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np


def normalize_opt_vars(opt_vars: Any) -> tuple[float, ...]:
    """
    Normalize optimization variables into a tuple of floats.
    
    Args:
        opt_vars: Optimization variables in various formats (scalar, array, sequence).
        
    Returns:
        Tuple of float values.
        
    Raises:
        TypeError: If the input cannot be converted to a tuple of floats.
    """
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


def stack_state_matrix(values: Any) -> np.ndarray:
    """
    Stack and reshape values into a 2D state matrix.
    
    Args:
        values: Input data (1D array, 2D array, 3D array, or sequence of arrays).
        
    Returns:
        2D numpy array with shape (n_samples, n_features).
    """
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


def ensure_callable_name(name: Any) -> Callable[..., str]:
    """
    Ensure a name is callable (for PySINDy library function names).
    
    Args:
        name: Either a callable or a string/value to wrap.
        
    Returns:
        A callable that returns the name as a string.
    """
    if callable(name):
        return name

    label = str(name)
    return lambda *_, label=label: label
