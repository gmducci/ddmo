from dataclasses import dataclass, field
from typing import Any, Optional, Literal, Type

from ddmo.test_function_factory.base import TestFunctionFamily  # abstract base


FormulationName = Literal["strong", "weak"]


def get_test_function_class(test_function_class: Any) -> Type[TestFunctionFamily]:
    """
    Normalize a test-function class specification.

    Accepts either a string (registered name) or a class object.
    """
    # string → look up
    if isinstance(test_function_class, str):
        name = test_function_class.lower()
        if name == "hannwindow":
            from ddmo.test_function_factory.hann_window import HannWindow
            return HannWindow
        raise ValueError(f"Unknown test function: {test_function_class!r}")

    # already a class → just return, but enforce subclass if desired
    if not issubclass(test_function_class, TestFunctionFamily):
        raise TypeError(
            f"test_functions_class must inherit from TestFunctionFamily, "
            f"got {test_function_class!r}"
        )
    return test_function_class


@dataclass
class FormulationSettings:
    """
    Settings for the formulation of the identification problem.
    """
    formulation_name: FormulationName = "strong"
    test_functions_class: Optional[Type[TestFunctionFamily]] = None
    formulation_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize test_functions_class: allow passing a string or a class
        if self.test_functions_class is not None:
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
            test_functions_class=test_functions_class,
            formulation_parameters=formulation_parameters,
        )