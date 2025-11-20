from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class SolverSettings:
    """
    Configuration settings for calling a solver backend.

    Attributes:
        solver_name (str): Identifier string for the solver (e.g. 'pysindy').
        optimizer (Optional[str]): Optional name of the optimizer within the solver.
        solver_parameters (dict[str, Any]): Keyword arguments passed to the solver.
        test_functions_class (Optional[Any]): Optional class for test functions.
    """
    solver_name: str
    regularization: Optional[str] = None
    regularization_parameters: dict[str, Any] = field(default_factory=dict)
    formulation: Optional[str] = "strong"
    test_functions_class: Optional[Any] = None

    def update(self, **kwargs) -> None:
        """Update internal solver_parameters dictionary with new values."""
        self.solver_parameters.update(kwargs)
        
    def __post_init__(self) -> None:
        """Normalize formulation names and enforce weak-form requirements."""
        if self.formulation is None:
            self.formulation = "strong"

        normalized = self.formulation.lower()
        if normalized not in {"strong", "weak"}:
            msg = f"Unknown formulation '{self.formulation}'. Choose 'strong' or 'weak'."
            raise ValueError(msg)

        self.formulation = normalized
        if normalized == "weak" and self.test_functions_class is None:
            msg = "Weak formulation requires 'test_functions_class'."
            raise ValueError(msg)
