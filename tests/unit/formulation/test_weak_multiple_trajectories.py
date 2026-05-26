import numpy as np

from ddmo.formulation.weak import WeakFormulation
from ddmo.library_functions import LibraryBuilder
from ddmo.utils.data_constructor import DataConstructor


def _make_data(x_train, time, library, x0=None):
    return DataConstructor(
        x_train=x_train,
        time_array=time,
        library_object=library.build,
        x_0=x0 or [np.asarray(x_train)[0, 0]],
        x_raw=x_train,
    )


def _make_formulation(data):
    return WeakFormulation(
        data_instance=data,
        class_attributes={
            "test_functions_class": "PolynomialWindow",
            "degree": 2,
            "window_length": 3,
            "stride": 1,
        },
    )


def test_weak_formulation_integrates_multiple_trajectories_independently():
    time = np.arange(5, dtype=float)
    library = LibraryBuilder().add_polynomial(max_degree=1, include_constant=True)
    x_one = np.arange(5, dtype=float).reshape(-1, 1)
    x_two = (100.0 + np.arange(5, dtype=float)).reshape(-1, 1)
    x_train = np.stack([x_one, x_two], axis=0)

    multi_data = _make_data(x_train, time, library, x0=[x_one[0], x_two[0]])
    first_data = _make_data(x_one, time, library, x0=[x_one[0]])
    second_data = _make_data(x_two, time, library, x0=[x_two[0]])

    multi_formulation = _make_formulation(multi_data)
    first_formulation = _make_formulation(first_data)
    second_formulation = _make_formulation(second_data)

    multi_lhs = multi_formulation.get_lhs(
        multi_data,
        [],
        opt_param_dict={},
    )
    expected_lhs = np.vstack(
        [
            first_formulation.get_lhs(first_data, [], opt_param_dict={}),
            second_formulation.get_lhs(second_data, [], opt_param_dict={}),
        ]
    )

    np.testing.assert_allclose(multi_lhs, expected_lhs)

    multi_rhs = multi_formulation.get_rhs(
        multi_data,
        [],
        opt_param_dict={},
    )["theta"]
    expected_rhs = np.vstack(
        [
            first_formulation.get_rhs(first_data, [], opt_param_dict={})["theta"],
            second_formulation.get_rhs(second_data, [], opt_param_dict={})["theta"],
        ]
    )

    np.testing.assert_allclose(multi_rhs, expected_rhs)
