"""Simple tests for StrongFormulation.get_lhs and get_rhs."""

import numpy as np
import pytest

from ddmo.formulation.strong import StrongFormulation
from ddmo.utils.data_constructor import DataConstructor


@pytest.fixture
def simple_data_instance():
    """A minimal DataConstructor for get_lhs / get_rhs tests."""
    x_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    x_dot = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])

    def library_object(params):
        return [
            (lambda x: x, "x"),
            (lambda x: x**2, "x_sq"),
        ]

    return DataConstructor(
        x_train=x_train,
        time_array=np.array([0.0, 1.0, 2.0]),
        library_object=library_object,
        x_0=[x_train[0]],
        x_raw=x_train,
        x_dot=x_dot,
    )


def test_get_lhs_falls_back_to_x_dot(simple_data_instance):
    formulation = StrongFormulation()
    result = formulation.get_lhs(simple_data_instance)

    assert np.allclose(result, simple_data_instance.x_dot)


def test_get_lhs_uses_custom_builder(simple_data_instance):
    formulation = StrongFormulation(lhs_builder=lambda *args, **kwargs: np.array([[9.0, 9.0]]))
    result = formulation.get_lhs(simple_data_instance)

    assert np.array_equal(result, np.array([[9.0, 9.0]]))


def test_get_lhs_requires_data_instance():
    formulation = StrongFormulation()

    with pytest.raises(ValueError, match="requires a data_instance"):
        formulation.get_lhs()


def test_get_rhs_returns_theta_and_library(simple_data_instance):
    formulation = StrongFormulation()
    result = formulation.get_rhs(simple_data_instance, (1.0,))

    assert isinstance(result, dict)
    assert "theta" in result
    assert "library" in result
    assert result["theta"].shape[0] == simple_data_instance.x_train.shape[0]


def test_get_rhs_uses_rhs_builder(simple_data_instance):
    formulation = StrongFormulation(rhs_builder=lambda *args, **kwargs: {"ok": True})
    result = formulation.get_rhs(simple_data_instance, (1.0,))

    assert result == {"ok": True}


def test_get_rhs_requires_opt_vars(simple_data_instance):
    formulation = StrongFormulation()

    with pytest.raises(ValueError, match="requires optimization variables"):
        formulation.get_rhs(simple_data_instance, None)
