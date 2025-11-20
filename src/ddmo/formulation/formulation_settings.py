from dataclasses import dataclass, field
from typing import Any, Optional, Literal


FormulationName = Literal["strong", "weak"]


def get_test_function_class(test_function_class: str):
    """
    Get test function class by name.
    """
    if test_function_class.lower() == "hannwindow":
        from ddmo.test_function_factory.hann_window import HannWindow
        return HannWindow
    else:
        raise ValueError(f"Unknown test function: {test_function_class}")


@dataclass
class FormulationSettings:
    """
    Settings for the formulation of the identification problem.
    """
    formulation_name: FormulationName = "strong"
    test_functions_class: Optional[Any] = None
    formulation_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize test_functions_class: allow passing a string
        if isinstance(self.test_functions_class, str):
            self.test_functions_class = get_test_function_class(
                self.test_functions_class
            )

        # Enforce that weak formulation must have test_functions_class
        if self.formulation_name == "weak" and self.test_functions_class is None:
            raise ValueError(
                "For 'weak' formulation, 'test_functions_class' must be provided."
            )

        # Ignore test_functions_class for strong formulation
        if self.formulation_name == "strong":
            self.test_functions_class = None

    def update(self, **kwargs) -> None:
        """Update internal formulation_parameters dictionary with new values."""
        self.formulation_parameters.update(kwargs)
        
    # Optional convenience constructors
    @classmethod
    def strong(cls, **formulation_parameters: Any) -> "FormulationSettings":
        return cls(
            formulation_name="strong",
            test_functions_class=None,
            formulation_parameters=None,
        )

    @classmethod
    def weak(
        cls,
        test_functions_class: Any,
        **formulation_parameters: Any,
    ) -> "FormulationSettings":
        return cls(
            formulation_name="weak",
            test_functions_class=get_test_function_class(test_functions_class),
            formulation_parameters=formulation_parameters,
        )