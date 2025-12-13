# ddmo/formulation/formulation_settings.py
from dataclasses import dataclass
from typing import Any


@dataclass
class FormulationSettings:
    """
    Configuration for a problem formulation (weak or strong).

    For strong formulation, only `formulation` is required:
        FormulationSettings(formulation="strong")

    For weak formulation, specify test function parameters:
        FormulationSettings(
            formulation="weak",
            test_function="PolynomialWindow",
            degree=3,
            window_length=51,
            stride=5
        )

    Attributes:
        formulation: Formulation type ("weak" or "strong").
        test_function: Test function family name (weak only).
        degree: Polynomial degree for test functions (weak only).
        window_length: Window length for test functions (weak only).
        stride: Stride between windows (weak only).
    """

    formulation: str
    test_function: str | None = None
    degree: int | None = None
    sigma: int | None = None
    window_length: int | None = None
    stride: int | None = None

    def __post_init__(self):
        """Validate formulation name."""
        valid = {"weak", "strong"}
        if self.formulation.lower() not in valid:
            raise ValueError(
                f"Unknown formulation '{self.formulation}'. Choose from: {valid}"
            )
        self.formulation = self.formulation.lower()

        # Validate weak formulation has required parameters
        if self.formulation == "weak" and self.test_function is None:
            raise ValueError("Weak formulation requires 'test_function' parameter.")

    def to_attributes_dict(self) -> dict[str, Any]:
        """Convert to dict format for backward compatibility with formulation classes."""
        attrs = {}
        if self.test_function is not None:
            attrs["test_functions_class"] = self.test_function
        if self.degree is not None:
            attrs["degree"] = self.degree
        if self.window_length is not None:
            attrs["window_length"] = self.window_length
        if self.stride is not None:
            attrs["stride"] = self.stride
        if self.sigma is not None:
            attrs["sigma"] = self.sigma
        return attrs
