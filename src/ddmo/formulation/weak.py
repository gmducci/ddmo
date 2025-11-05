from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .base import Formulation

IntegralSolver = Callable[..., np.ndarray]


@dataclass
class WeakFormulation(Formulation):
    """
    Weak formulation helper that delegates integral evaluation to a solver.

    The default behaviour mirrors the sketch in the design notes: both sides of
    the equation use the same integral, but the left-hand side carries a minus
    sign.
    """

    integral_solver: IntegralSolver | None = None

    def solve_integral(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if self.integral_solver is None:
            msg = "An integral_solver callable must be provided for weak formulations."
            raise ValueError(msg)
        return self.integral_solver(*args, **kwargs)

    def get_lhs(self, *args: Any, **kwargs: Any) -> np.ndarray:
        integral = self.solve_integral(*args, **kwargs)
        return -integral

    def get_rhs(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.solve_integral(*args, **kwargs)
