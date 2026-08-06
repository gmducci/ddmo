import numpy as np
import pytest

from ddmo.utils.constructors import IdentificationResult
from ddmo.utils.symbolic_equations import (
    format_symbolic_equations,
    symbolic_equations,
)


class FakeLibrary:
    def __init__(self, names):
        self.names = names

    def get_feature_names(self, input_features=None):
        if input_features is None:
            return self.names
        return [name.format(*input_features) for name in self.names]


class FakeModel:
    def __init__(self, coefficients, names, active_mask=None):
        self._coefficients = np.asarray(coefficients, dtype=float)
        self.feature_library = FakeLibrary(names)
        self.active_mask = active_mask

    def coefficients(self):
        return self._coefficients


def test_format_symbolic_equations_formats_sparse_system():
    equations = format_symbolic_equations(
        np.array([[1.25, 0.0], [-2.0, 3.0], [1e-14, 4.0]]),
        ["1", "x", "y"],
        state_names=["x", "y"],
        precision=3,
    )

    assert equations == [
        "dx/dt = 1.25 - 2*x",
        "dy/dt = 3*x + 4*y",
    ]


def test_symbolic_equations_reads_model_feature_names():
    model = FakeModel(
        coefficients=np.array([[1.0], [-0.5]]),
        names=["1", "{0}"],
    )

    equations = symbolic_equations(
        model,
        input_features=["population"],
        state_names=["population"],
    )

    assert equations == ["dpopulation/dt = 1 - 0.5*population"]


def test_symbolic_equations_reads_identification_result_index():
    result = IdentificationResult(
        problem=None,
        result=None,
        objective_space=None,
        solutions=[],
        coeff_matrices=[],
        alphas=[],
        models=[
            FakeModel(np.array([[1.0]]), ["x0"]),
            FakeModel(np.array([[2.0]]), ["x0"]),
        ],
        theta_matrices=[],
    )

    assert result.symbolic_equations(index=1) == ["dx_0/dt = 2*x0"]


def test_symbolic_equations_applies_active_mask():
    model = FakeModel(
        coefficients=np.array([[2.0], [3.0]]),
        names=["x0", "bad", "x1"],
        active_mask=np.array([True, False, True]),
    )

    equations = symbolic_equations(model, state_names=["x"])

    assert equations == ["dx/dt = 2*x0 + 3*x1"]


def test_format_symbolic_equations_validates_shapes():
    with pytest.raises(ValueError, match="feature_names"):
        format_symbolic_equations(np.ones((2, 1)), ["x0"])
