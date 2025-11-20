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
    formulation: Optional[str] = 'strong'
    test_functions_class: Optional[Any] = None

    def update(self, **kwargs) -> None:
        """Update internal solver_parameters dictionary with new values."""
        self.solver_parameters.update(kwargs)
