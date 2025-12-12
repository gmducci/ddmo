from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolverSettings:
    """
    Configuration for a sparse regression solver.

    Attributes:
        solver_name: Identifier for the solver backend (e.g., "pysindy", "scikit").
        regularization: Regularization method (e.g., "STLSQ", "Lasso", "Ridge").
        regularization_parameters: Keyword arguments for the regularization method.

    Example:
        SolverSettings(
            solver_name="pysindy",
            regularization="STLSQ",
            regularization_parameters={"alpha": 0.15, "threshold": 0.2}
        )
    """

    solver_name: str
    regularization: str | None = None
    regularization_parameters: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs) -> None:
        """Update regularization_parameters with new values."""
        self.regularization_parameters.update(kwargs)
