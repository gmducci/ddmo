import numpy as np

from ddmo.optimizer.objective_functions import compute_mse
from ddmo.utils.data_constructor import DataConstructor


def test_compute_mse_slices_stacked_solution_by_trajectory_length():
    x_raw = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[10.0], [20.0], [30.0]],
        ]
    )
    sol = np.vstack(
        [
            x_raw[0],
            x_raw[1] + 2.0,
        ]
    )
    data = DataConstructor(
        x_train=x_raw,
        time_array=np.arange(3, dtype=float),
        library_object=lambda _: [],
        x_0=[x_raw[0, 0], x_raw[1, 0]],
        x_raw=x_raw,
    )

    assert compute_mse(data, sol) == 4.0
