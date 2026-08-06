"""Formatting utilities for symbolic discovered equations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def symbolic_equations(
    result_or_model: Any,
    *,
    index: int = 0,
    input_features: Sequence[str] | None = None,
    state_names: Sequence[str] | None = None,
    precision: int = 6,
    coefficient_threshold: float = 1e-12,
) -> list[str]:
    """Return formatted equations from an identification result or fitted model.

    Parameters
    ----------
    result_or_model:
        Either an ``IdentificationResult`` returned by ``identify`` or a fitted
        model object with ``coefficients()`` and ``feature_library`` attributes.
    index:
        Pareto/result index to inspect when ``result_or_model`` is an
        ``IdentificationResult``.
    input_features:
        Names used by the feature library for state variables, such as
        ``["x", "y", "z"]``.
    state_names:
        Names used on the left-hand side. Defaults to ``input_features``.
    precision:
        Significant digits used for coefficients.
    coefficient_threshold:
        Coefficients with absolute value at or below this value are omitted.
    """
    model = _select_model(result_or_model, index)
    coefficients = _coefficients_from_model(model)
    feature_names = _feature_names_from_model(model, input_features)

    return format_symbolic_equations(
        coefficients,
        feature_names,
        state_names=state_names or input_features,
        precision=precision,
        coefficient_threshold=coefficient_threshold,
    )


def format_symbolic_equations(
    coefficients: np.ndarray,
    feature_names: Sequence[str],
    *,
    state_names: Sequence[str] | None = None,
    precision: int = 6,
    coefficient_threshold: float = 1e-12,
) -> list[str]:
    """Format a coefficient matrix and feature names as ODE equations."""
    coeff_array = np.asarray(coefficients, dtype=float)
    if coeff_array.ndim != 2:
        raise ValueError("coefficients must be a two-dimensional array.")
    if coeff_array.shape[0] != len(feature_names):
        raise ValueError("feature_names must match the number of coefficient rows.")
    if coefficient_threshold < 0:
        raise ValueError("coefficient_threshold must be non-negative.")

    n_states = coeff_array.shape[1]
    lhs_names = _state_names(state_names, n_states)
    equations = []

    for state_idx, state_name in enumerate(lhs_names):
        rhs = _format_rhs(
            coeff_array[:, state_idx],
            feature_names,
            precision=precision,
            coefficient_threshold=coefficient_threshold,
        )
        equations.append(f"d{state_name}/dt = {rhs}")

    return equations


def print_symbolic_equations(
    result_or_model: Any,
    *,
    index: int = 0,
    input_features: Sequence[str] | None = None,
    state_names: Sequence[str] | None = None,
    precision: int = 6,
    coefficient_threshold: float = 1e-12,
) -> None:
    """Print formatted symbolic equations from an identification result or model."""
    for equation in symbolic_equations(
        result_or_model,
        index=index,
        input_features=input_features,
        state_names=state_names,
        precision=precision,
        coefficient_threshold=coefficient_threshold,
    ):
        print(equation)


def _select_model(result_or_model: Any, index: int) -> Any:
    if hasattr(result_or_model, "models"):
        models = result_or_model.models
        if not 0 <= index < len(models):
            raise IndexError(f"index must be in [0, {len(models) - 1}].")
        model = models[index]
    else:
        model = result_or_model

    if model is None:
        raise ValueError("No fitted model is available for symbolic formatting.")
    return model


def _coefficients_from_model(model: Any) -> np.ndarray:
    if hasattr(model, "coefficients") and callable(model.coefficients):
        coefficients = model.coefficients()
    elif hasattr(model, "_coefficients"):
        coefficients = model._coefficients
    else:
        raise ValueError("model must provide coefficients.")

    if coefficients is None:
        raise ValueError("model coefficients are not available.")
    return np.asarray(coefficients, dtype=float)


def _feature_names_from_model(
    model: Any,
    input_features: Sequence[str] | None,
) -> list[str]:
    if not hasattr(model, "feature_library"):
        raise ValueError("model must provide a feature_library.")

    library = model.feature_library
    if not hasattr(library, "get_feature_names"):
        raise ValueError("model.feature_library must provide get_feature_names().")

    feature_names = list(library.get_feature_names(input_features))
    active_mask = getattr(model, "active_mask", None)
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape[0] != len(feature_names):
            raise ValueError("model.active_mask does not match feature names.")
        feature_names = [
            name for name, active in zip(feature_names, active_mask) if active
        ]

    return feature_names


def _state_names(state_names: Sequence[str] | None, n_states: int) -> list[str]:
    if state_names is None:
        return [f"x_{i}" for i in range(n_states)]
    names = list(state_names)
    if len(names) != n_states:
        raise ValueError("state_names must match the number of state variables.")
    return names


def _format_rhs(
    coefficients: np.ndarray,
    feature_names: Sequence[str],
    *,
    precision: int,
    coefficient_threshold: float,
) -> str:
    terms = []
    for coefficient, feature_name in zip(coefficients, feature_names):
        if abs(coefficient) <= coefficient_threshold:
            continue
        terms.append(_format_term(coefficient, feature_name, precision))

    if not terms:
        return "0"

    rhs = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            rhs += f" - {term[1:]}"
        else:
            rhs += f" + {term}"
    return rhs


def _format_term(coefficient: float, feature_name: str, precision: int) -> str:
    coefficient_text = f"{coefficient:.{precision}g}"
    if feature_name == "1":
        return coefficient_text
    return f"{coefficient_text}*{feature_name}"
