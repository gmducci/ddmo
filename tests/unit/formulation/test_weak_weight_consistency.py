import numpy as np
import pytest

from ddmo.formulation.weak import WeakFormulation, _build_consistent_weak_weights
from ddmo.library_functions import LibraryBuilder
from ddmo.utils.data_constructor import DataConstructor


def _data(time):
    state = time[:, None]
    return DataConstructor(
        x_train=state, time_array=time, x_0=[state[0]], x_raw=state
    )


def test_linear_state_and_test_interpolants_satisfy_weak_identity():
    # Nonzero endpoint values exercise the boundary term in integration by parts.
    phi = np.array([2.0, 1.0, 3.0])
    lhs, rhs = _build_consistent_weak_weights(phi, dt=0.5)
    state = np.array([4.0, 4.5, 5.0])  # x_dot = 1

    np.testing.assert_allclose(state @ lhs, np.ones(3) @ rhs, atol=1e-14)
    np.testing.assert_allclose(state @ lhs, 1.75, atol=1e-14)
    np.testing.assert_allclose(np.ones(3) @ lhs, 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "family_parameters",
    [
        {"test_functions_class": "PolynomialWindow", "degree": 2},
        {"test_functions_class": "HannWindow"},
        {"test_functions_class": "GaussianWindow", "sigma": 1.0},
    ],
)
def test_formulation_matches_constant_vector_field(family_parameters):
    time = np.arange(7, dtype=float)
    data = _data(time)
    library = LibraryBuilder().add_polynomial(max_degree=0, include_constant=True)
    formulation = WeakFormulation(
        data_instance=data,
        class_attributes={**family_parameters, "window_length": 3, "stride": 1},
        library_object=library.build,
    )

    lhs = formulation.get_lhs(data, [], opt_param_dict={})
    rhs = formulation.get_rhs(data, [], opt_param_dict={})["theta"]
    np.testing.assert_allclose(lhs, rhs, atol=1e-14)


@pytest.mark.parametrize("time", [np.array([0.0]), np.array([0.0, 1.0, 2.1]), np.array([1.0, 0.0, -1.0])])
def test_weak_formulation_requires_a_uniform_increasing_grid(time):
    data = _data(time)
    with pytest.raises(ValueError, match="time_array"):
        WeakFormulation(
            data_instance=data,
            class_attributes={
                "test_functions_class": "HannWindow",
                "window_length": 3,
                "stride": 1,
            },
        )
