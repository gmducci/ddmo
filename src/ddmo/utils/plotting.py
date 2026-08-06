from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ddmo.test_function_factory.base import TestFunctionFamily
from ddmo.utils.ode_data_generator import ODEDataBatch


def split_stacked_trajectories(
    trajectory: np.ndarray | None,
    n_trajectories: int,
    n_time: int,
) -> list[np.ndarray | None]:
    """Split a stacked trajectory array into one block per initial condition."""
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1.")
    if n_time < 1:
        raise ValueError("n_time must be at least 1.")
    if trajectory is None:
        return [None] * n_trajectories

    trajectory_array = np.asarray(trajectory)
    expected_rows = n_trajectories * n_time
    if trajectory_array.shape[0] < expected_rows:
        raise ValueError(
            "trajectory does not contain enough rows for the requested split."
        )

    return [
        trajectory_array[i * n_time : (i + 1) * n_time]
        for i in range(n_trajectories)
    ]


def plot_ode_data_batch(
    batch: ODEDataBatch,
    *,
    solution: np.ndarray | None = None,
    variable_names: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Generated ODE batch",
    show: bool = True,
) -> tuple[Any, np.ndarray]:
    """Plot clean, noisy, and optional identified trajectories from an ODE batch."""
    n_trajectories, _, n_variables = batch.x_noisy.shape
    if variable_names is None:
        labels = [f"x_{i}" for i in range(n_variables)]
    else:
        labels = list(variable_names)
        if len(labels) != n_variables:
            raise ValueError("variable_names must match the number of variables.")

    if figsize is None:
        figsize = (10.0, max(2.6 * n_variables, 4.0))
    figure, axes = plt.subplots(n_variables, 1, figsize=figsize, sharex=True)
    plot_axes = np.asarray(axes, dtype=object).reshape(-1)
    predicted_blocks = split_stacked_trajectories(
        solution,
        n_trajectories=n_trajectories,
        n_time=len(batch.time),
    )

    for var_idx, name in enumerate(labels):
        axis = plot_axes[var_idx]
        for traj_idx in range(n_trajectories):
            axis.plot(
                batch.time,
                batch.x_clean[traj_idx, :, var_idx],
                color="black",
                linewidth=1.5,
                alpha=0.8,
                label=f"clean {name}" if traj_idx == 0 else None,
            )
            axis.scatter(
                batch.time,
                batch.x_noisy[traj_idx, :, var_idx],
                s=5,
                alpha=0.2,
                label=f"noisy {name}" if traj_idx == 0 else None,
            )

            predicted = predicted_blocks[traj_idx]
            if predicted is not None:
                n = min(len(batch.time), len(predicted))
                axis.plot(
                    batch.time[:n],
                    predicted[:n, var_idx],
                    linestyle="--",
                    linewidth=1.5,
                    label=f"identified {name}" if traj_idx == 0 else None,
                )

        axis.set_ylabel(name)
        axis.legend(frameon=False)

    plot_axes[-1].set_xlabel("time")
    figure.suptitle(title)
    figure.tight_layout()
    if show:
        plt.show()
    return figure, plot_axes


def plot_test_function_parameter_comparison(
    family_class: type[TestFunctionFamily],
    time: Sequence[float] | np.ndarray,
    *,
    window_lengths: Sequence[int],
    strides: Sequence[int],
    family_params: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    show: bool = True,
) -> tuple[Any, np.ndarray]:
    """Compare one test-function family across window lengths and strides.

    Rows vary the window length and columns vary the stride. Each panel shows
    every translated test function produced by that parameter combination.

    Parameters
    ----------
    family_class
        Test-function family class, such as ``GaussianWindow`` or ``HannWindow``.
    time
        Uniform physical time grid on which the test functions are built.
    window_lengths
        Odd window lengths to compare.
    strides
        Positive strides to compare.
    family_params
        Additional parameters required by the family, such as ``sigma`` for a
        Gaussian window or ``degree`` for a polynomial window.
    figsize
        Figure size. By default, it scales with the subplot grid.
    cmap
        Matplotlib colormap used to distinguish translated test functions.
    show
        Display the completed figure with ``matplotlib.pyplot.show``.

    Returns
    -------
    figure, axes
        The Matplotlib figure and a two-dimensional array of axes.
    """
    time_array, dt = _validate_time(time)
    lengths = [int(length) for length in window_lengths]
    stride_values = [int(stride) for stride in strides]

    if not lengths:
        raise ValueError("window_lengths must contain at least one value.")
    if not stride_values:
        raise ValueError("strides must contain at least one value.")
    if any(length < 3 or length % 2 == 0 for length in lengths):
        raise ValueError("window_lengths must contain odd integers of at least 3.")
    if any(stride < 1 for stride in stride_values):
        raise ValueError("strides must contain positive integers.")

    if figsize is None:
        figsize = (3.6 * len(stride_values), 2.8 * len(lengths))
    figure, axes = plt.subplots(
        len(lengths),
        len(stride_values),
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extra_params = dict(family_params or {})

    for row, window_length in enumerate(lengths):
        for column, stride in enumerate(stride_values):
            family = family_class(
                dt=dt,
                window_length=window_length,
                stride=stride,
                **extra_params,
            )
            tests = np.asarray(family.build_tests((time_array.size,)), dtype=float)
            colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, tests.shape[0]))
            axis = axes[row, column]

            for test, color in zip(tests, colors):
                axis.plot(time_array, test, color=color, linewidth=1.2, alpha=0.8)

            axis.set_title(f"window length = {window_length}, stride = {stride}")
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0)
            if row == len(lengths) - 1:
                axis.set_xlabel(r"$t$")
            if column == 0:
                axis.set_ylabel(r"$\phi_k(t)$")

    figure.suptitle(f"{family_class.__name__}: window length and stride comparison")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


def plot_test_function_family(
    family: TestFunctionFamily,
    time: Sequence[float] | np.ndarray,
    *,
    n_functions: int = 5,
    representative_index: int | None = None,
    axes: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (7.0, 5.5),
    cmap: str = "viridis",
    show: bool = True,
) -> tuple[Any, np.ndarray]:
    """Plot translated test functions and one function with its derivative.

    The first panel illustrates how members of a test-function family cover the
    time domain. The second panel shows a representative member and its time
    derivative, as used on the two sides of the weak formulation.

    Parameters
    ----------
    family
        Instantiated one-dimensional test-function family.
    time
        Physical time grid on which the translated functions are built.
    n_functions
        Maximum number of translated functions shown in the first panel.
        Functions are sampled evenly across all available windows.
    representative_index
        Index of the function shown in the second panel. By default, the
        central available function is used.
    axes
        Optional pair of Matplotlib axes to draw into.
    figsize
        Figure size used only when ``axes`` is not supplied.
    cmap
        Matplotlib colormap used for the translated functions.
    show
        Display the completed figure with ``matplotlib.pyplot.show``.

    Returns
    -------
    figure, axes
        The Matplotlib figure and a length-two array of axes.
    """
    time_array, dt = _validate_time(time)
    family_dt = np.asarray(family.dt, dtype=float)
    if family_dt.size != 1 or not np.isclose(float(family_dt), dt):
        raise ValueError("The spacing of time must match family.dt.")
    if n_functions < 1:
        raise ValueError("n_functions must be at least 1.")

    tests = np.asarray(family.build_tests((time_array.size,)), dtype=float)
    if tests.ndim != 2 or tests.shape[1] != time_array.size:
        raise ValueError("family.build_tests must return one-dimensional test functions.")
    if tests.shape[0] == 0:
        raise ValueError("The family did not produce any test functions.")

    plot_axes = _prepare_axes(axes, figsize)
    figure = plot_axes[0].figure

    shown_indices = np.unique(
        np.linspace(
            0,
            tests.shape[0] - 1,
            min(n_functions, tests.shape[0]),
            dtype=int,
        )
    )
    colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, shown_indices.size))

    for index, color in zip(shown_indices, colors):
        plot_axes[0].plot(
            time_array,
            tests[index],
            color=color,
            linewidth=1.6,
            label=rf"$\phi_{{{index}}}(t)$",
        )

    if representative_index is None:
        representative_index = tests.shape[0] // 2
    if not 0 <= representative_index < tests.shape[0]:
        raise IndexError(
            f"representative_index must be in [0, {tests.shape[0] - 1}]."
        )

    derivative = _translated_derivative(family, time_array.size, representative_index)
    plot_axes[1].plot(
        time_array,
        tests[representative_index],
        color="C0",
        linewidth=2.0,
        label=rf"$\phi_{{{representative_index}}}(t)$",
    )
    plot_axes[1].plot(
        time_array,
        derivative,
        color="C3",
        linewidth=1.8,
        linestyle="--",
        label=rf"$\dot{{\phi}}_{{{representative_index}}}(t)$",
    )

    plot_axes[0].set_title("Translated test functions")
    plot_axes[0].set_ylabel(r"$\phi_k(t)$")
    plot_axes[0].legend(frameon=False, ncols=2)

    plot_axes[1].set_title("Representative test function and derivative")
    plot_axes[1].set_xlabel(r"$t$")
    plot_axes[1].set_ylabel("value")
    plot_axes[1].axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    plot_axes[1].legend(frameon=False)

    for axis in plot_axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0)

    figure.tight_layout()
    if show:
        plt.show()
    return figure, plot_axes


def _validate_time(time: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float]:
    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("time must be a one-dimensional array with at least 2 points.")
    time_steps = np.diff(time_array)
    if not np.all(time_steps > 0.0):
        raise ValueError("time must be strictly increasing.")
    if not np.allclose(time_steps, time_steps[0]):
        raise ValueError("time must be uniformly spaced.")
    return time_array, float(time_steps[0])


def _prepare_axes(
    axes: Sequence[Any] | None, figsize: tuple[float, float]
) -> np.ndarray:
    if axes is None:
        _, created_axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        return np.asarray(created_axes)

    plot_axes = np.asarray(axes, dtype=object).reshape(-1)
    if plot_axes.size != 2:
        raise ValueError("axes must contain exactly two Matplotlib axes.")
    if plot_axes[0].figure is not plot_axes[1].figure:
        raise ValueError("axes must belong to the same Matplotlib figure.")
    return plot_axes


def _translated_derivative(
    family: TestFunctionFamily, grid_size: int, index: int
) -> np.ndarray:
    derivative = np.asarray(family.dphi, dtype=float)
    if derivative.ndim != 1 or derivative.size % 2 == 0:
        raise ValueError("family.dphi must be a one-dimensional array of odd length.")

    center = int(np.asarray(family.centers)[index])
    half = derivative.size // 2
    translated = np.zeros(grid_size, dtype=float)
    translated[center - half : center + half + 1] = derivative
    return translated


def _show_demo() -> None:
    """Display a Gaussian-family parameter comparison."""
    from ddmo.test_function_factory.gaussian_window import GaussianWindow

    time = np.linspace(0.0, 10.0, 501)
    plot_test_function_parameter_comparison(
        GaussianWindow,
        time,
        window_lengths=[51, 101],
        strides=[30, 75],
        family_params={"sigma": 15.0},
    )


if __name__ == "__main__":
    _show_demo()
