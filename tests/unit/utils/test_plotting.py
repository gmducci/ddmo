import matplotlib.pyplot as plt
import numpy as np
import pytest

from ddmo.test_function_factory.hann_window import HannWindow
from ddmo.utils.ode_data_generator import ODEDataBatch
from ddmo.utils.plotting import (
    plot_ode_data_batch,
    plot_test_function_family,
    plot_test_function_parameter_comparison,
    split_stacked_trajectories,
)


def test_plot_test_function_parameter_comparison():
    time = np.linspace(0.0, 2.0, 101)

    figure, axes = plot_test_function_parameter_comparison(
        HannWindow,
        time,
        window_lengths=[11, 21],
        strides=[10, 20],
        show=False,
    )

    assert axes.shape == (2, 2)
    assert len(axes[0, 0].lines) > len(axes[0, 1].lines)
    assert "window length = 21, stride = 20" == axes[1, 1].get_title()
    plt.close(figure)


@pytest.mark.parametrize(
    ("window_lengths", "strides", "message"),
    [
        ([], [1], "window_lengths"),
        ([10], [1], "odd integers"),
        ([11], [0], "positive integers"),
    ],
)
def test_plot_test_function_parameter_comparison_validates_parameters(
    window_lengths, strides, message
):
    with pytest.raises(ValueError, match=message):
        plot_test_function_parameter_comparison(
            HannWindow,
            np.arange(30, dtype=float),
            window_lengths=window_lengths,
            strides=strides,
            show=False,
        )


def test_plot_test_function_family_builds_didactical_panels(monkeypatch):
    time = np.linspace(0.0, 2.0, 101)
    family = HannWindow(dt=time[1] - time[0], window_length=21, stride=10)
    show_calls = []
    monkeypatch.setattr("ddmo.utils.plotting.plt.show", lambda: show_calls.append(True))

    figure, axes = plot_test_function_family(family, time, n_functions=4)

    assert figure is axes[0].figure
    assert axes.shape == (2,)
    assert len(axes[0].lines) == 4
    assert len(axes[1].lines) == 3  # phi, dphi, and the zero reference line
    assert axes[1].get_xlabel() == "$t$"
    assert show_calls == [True]
    plt.close(figure)


def test_plot_test_function_family_uses_supplied_axes():
    time = np.linspace(0.0, 1.0, 51)
    family = HannWindow(dt=time[1] - time[0], window_length=11, stride=5)
    figure, supplied_axes = plt.subplots(1, 2)

    returned_figure, returned_axes = plot_test_function_family(
        family, time, axes=supplied_axes, show=False
    )

    assert returned_figure is figure
    assert np.array_equal(returned_axes, supplied_axes)
    plt.close(figure)


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([0.0], "at least 2"),
        ([0.0, 1.0, 0.5], "strictly increasing"),
        ([0.0, 1.0, 3.0], "uniformly spaced"),
    ],
)
def test_plot_test_function_family_validates_time(time, message):
    family = HannWindow(dt=1.0, window_length=5, stride=1)

    with pytest.raises(ValueError, match=message):
        plot_test_function_family(family, time)


def test_plot_test_function_family_requires_matching_time_step():
    family = HannWindow(dt=0.5, window_length=5, stride=1)

    with pytest.raises(ValueError, match="match family.dt"):
        plot_test_function_family(family, np.arange(10, dtype=float))


def test_split_stacked_trajectories():
    trajectory = np.arange(12).reshape(6, 2)

    blocks = split_stacked_trajectories(
        trajectory,
        n_trajectories=3,
        n_time=2,
    )

    assert len(blocks) == 3
    np.testing.assert_array_equal(blocks[0], trajectory[:2])
    np.testing.assert_array_equal(blocks[2], trajectory[4:6])


def test_plot_ode_data_batch_with_solution():
    time = np.linspace(0.0, 1.0, 5)
    x_clean = np.stack(
        [
            np.column_stack([time, time**2]),
            np.column_stack([2.0 * time, 3.0 * time]),
        ],
        axis=0,
    )
    batch = ODEDataBatch(
        time=time,
        x_clean=x_clean,
        x_noisy=x_clean + 0.1,
        x_dot_clean=np.zeros_like(x_clean),
        initial_conditions=[x_clean[0, 0], x_clean[1, 0]],
    )
    solution = x_clean.reshape(-1, 2)

    figure, axes = plot_ode_data_batch(
        batch,
        solution=solution,
        variable_names=["x", "y"],
        show=False,
    )

    assert axes.shape == (2,)
    assert axes[0].get_ylabel() == "x"
    assert axes[1].get_xlabel() == "time"
    assert len(axes[0].lines) == 4
    plt.close(figure)
