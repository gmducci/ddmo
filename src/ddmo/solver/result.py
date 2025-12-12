"""Solver result dataclass for consistent return types."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class SolverResult:
    """
    Structured result from sparse regression solvers.

    Provides a consistent return type for all solvers, eliminating
    fragile tuple unpacking in the optimizer.

    Attributes:
        solution: Reconstructed trajectory (n_samples, n_vars) or None if failed.
        coefficients: Sparse regression coefficients (n_features, n_targets).
        opt_vars: Tuple of optimizer parameter values used.
        model: Fitted model object (for simulation).
        feature_matrix: Feature matrix (Theta) used in regression.
    """

    solution: Optional[np.ndarray]
    coefficients: Optional[np.ndarray]
    opt_vars: tuple[float, ...]
    model: Any
    feature_matrix: Optional[np.ndarray] = None

    @property
    def success(self) -> bool:
        """Whether the solver produced a valid solution."""
        return self.solution is not None and self.coefficients is not None
