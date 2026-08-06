# Quickstart

This page shows the minimal workflow for running `ddmo` from the repository checkout.

The goal is not to cover every option, but to make the main user-facing objects clear enough to run and modify an existing identification workflow.

## 1. Install the package

From the repository root:

```bash
pip install -e .
```

For development, install the optional development dependencies if they are configured in `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

You can check that the package imports correctly with:

```bash
python -c "import ddmo; print(ddmo.__all__)"
```

## 2. Run an existing example

The fastest way to verify that the package works is to run one of the tutorial examples:

```bash
python examples/quickstart/goodwin_weak.py
```

or:

```bash
python examples/quickstart/brusselator_custom_library.py
```

These examples load or generate trajectory data, build a candidate function library, configure the formulation, solver, and optimizer, and run model identification.

## 3. Import the public API

The most common user-facing objects are available directly from `ddmo`:

```python
from ddmo import (
    BENCHMARK_FOLDER,
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.library_functions import LibraryBuilder
```

The main workflow is built from these components:

- `DataConstructor`: stores the trajectory data used for identification;
- `LibraryBuilder`: defines the candidate model terms;
- `FormulationSettings`: selects the strong or weak formulation;
- `SolverSettings`: configures the sparse-regression backend;
- `OptimizerSettings`: defines the parameter-search problem;
- `identify`: runs the identification pipeline.

## 4. Load benchmark data

The examples use benchmark data included with the package:

```python
import numpy as np

x_0_train = np.load(BENCHMARK_FOLDER / "x_0.npy")
x_raw = np.load(BENCHMARK_FOLDER / "x_train_noise.npy")
x_train = np.load(BENCHMARK_FOLDER / "x_train.npy")
x_dot_hat = np.load(BENCHMARK_FOLDER / "x_dot_train.npy")
time_array = np.load(BENCHMARK_FOLDER / "time.npy")
```

The arrays have different roles:

- `x_raw`: noisy trajectory data;
- `x_train`: training trajectory data;
- `x_dot_hat`: derivative data used by the strong formulation;
- `time_array`: time points;
- `x_0_train`: initial condition.

## 5. Define a candidate library

A candidate library defines the terms that may appear in the discovered model.

```python
library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2, include_interactions=True)
    .add_hill(exponent_param="gamma_1")
    .add_exponential(rate_param="gamma_2")
)
```

Here, `gamma_1` and `gamma_2` are library parameters. They can later be optimized through `OptimizerSettings.library_bounds`.

## 6. Construct the data object

The data object is passed to the identification pipeline.

For a weak-form workflow, derivative data are not required:

```python
skip = 10

training_data = DataConstructor(
    x_train=x_raw[::skip, :],
    time_array=time_array[::skip],
    x_0=[x_0_train],
    x_raw=x_raw[::skip, :],
)
```

For a strong-form workflow, derivative data should be provided:

```python
training_data = DataConstructor(
    x_train=x_train[20:-20, :],
    time_array=time_array[20:-20],
    x_dot=x_dot_hat[20:-20, :],
    x_0=[x_0_train],
    x_raw=x_raw[20:-20, :],
)
```

## 7. Configure the formulation

The formulation controls how the regression problem is constructed.

For the weak formulation:

```python
formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=65,
    stride=50,
)
```

For the strong formulation:

```python
formulation_settings = FormulationSettings(
    formulation="strong",
)
```

Use the strong formulation when reliable derivative information is available. Use the weak formulation when the workflow should rely on integrated quantities instead of pointwise derivatives.

## 8. Configure the solver

The solver estimates sparse model coefficients once the regression problem has been built.

```python
solver_settings = SolverSettings(
    solver_name="pysindy",
    regularization="STLSQ",
    regularization_parameters={
        "normalize_columns": False,
        "alpha": 0.05,
        "threshold": 0.1,
    },
    scale_features=True,
)
```

In the current PySINDy backend:

- `solver_name="pysindy"` selects the PySINDy-based solver;
- `regularization="STLSQ"` selects the sparse-regression optimizer;
- `regularization_parameters` are passed to the selected optimizer;
- `scale_features=True` applies DDMO-level scaling before regression and transforms coefficients back afterward.

`scale_features` is separate from optimizer-specific options such as `normalize_columns`.

## 9. Configure the optimizer

The optimizer searches over selected parameters of the identification pipeline.

For a weak-form workflow, it is common to optimize library and formulation parameters:

```python
library_bounds = {
    "gamma_1": (8.4, 9.6),
    "gamma_2": (1.0, 10.0),
}

formulation_bounds = {
    "degree": (2, 7),
    "stride": (50, 250),
    "window_length": (127, 801),
}

optimizer_settings = OptimizerSettings(
    library_bounds=library_bounds,
    formulation_bounds=formulation_bounds,
    n_gen=10,
    algorithm="NSGA3",
    pop_size=50,
)
```

For a strong-form workflow, formulation bounds may be empty and solver parameters can be optimized instead:

```python
optimizer_settings = OptimizerSettings(
    library_bounds=library_bounds,
    formulation_bounds={},
    solver_bounds={"threshold": (0.01, 1.0)},
    n_gen=3,
    algorithm="NSGA3",
    pop_size=50,
)
```

Bounds use the format:

```python
{"parameter_name": (lower_bound, upper_bound)}
```

For weak formulations, `window_length` always means the actual number of data
samples in the window. Optimized values are rounded to an integer and even
values are advanced to the next valid odd length.

## 10. Run identification

Call `identify` with the configured objects:

```python
result = identify(
    data_instance=training_data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
)
```

The default objectives are trajectory MSE and Durbin-Watson deviation. To use
trajectory error and optional model sparsity instead:

```python
from ddmo.optimizer import make_active_coefficient_count, trajectory_mse

objectives = {
    "trajectory_mse": trajectory_mse,
    "active_coefficients": make_active_coefficient_count(tolerance=1e-6),
}

result = identify(
    data_instance=training_data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
    objective_functions=objectives,
)
```

See [Optimization objectives](objectives.md) for custom objective functions.

The returned object contains the identification results, including fitted trajectories and optimization outputs.

For example, the existing scripts access:

```python
print(result.alphas)
```

and plot the first fitted solution against the data:

```python
import matplotlib.pyplot as plt

plt.plot(result.solutions[0])
plt.plot(training_data.x_raw, alpha=0.5)
plt.show()
```

## 11. Minimal weak-form script

The following script combines the main pieces into a single weak-form workflow:

```python
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ddmo import (
    BENCHMARK_FOLDER,
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.library_functions import LibraryBuilder

warnings.filterwarnings("ignore")

x_0_train = np.load(BENCHMARK_FOLDER / "x_0.npy")
x_raw = np.load(BENCHMARK_FOLDER / "x_train_noise.npy")
time_array = np.load(BENCHMARK_FOLDER / "time.npy")

library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2, include_interactions=True)
    .add_hill(exponent_param="gamma_1")
    .add_exponential(rate_param="gamma_2")
)

skip = 10
training_data = DataConstructor(
    x_train=x_raw[::skip, :],
    time_array=time_array[::skip],
    x_0=[x_0_train],
    x_raw=x_raw[::skip, :],
)

formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=65,
    stride=50,
)

solver_settings = SolverSettings(
    solver_name="pysindy",
    regularization="STLSQ",
    regularization_parameters={
        "normalize_columns": False,
        "alpha": 0.05,
        "threshold": 0.1,
    },
    scale_features=True,
)

optimizer_settings = OptimizerSettings(
    library_bounds={
        "gamma_1": (8.4, 9.6),
        "gamma_2": (1.0, 10.0),
    },
    formulation_bounds={
        "degree": (2, 7),
        "stride": (50, 250),
        "window_length": (127, 801),
    },
    n_gen=10,
    algorithm="NSGA3",
    pop_size=50,
)

result = identify(
    data_instance=training_data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
)

print(result.alphas)

plt.plot(result.solutions[0])
plt.plot(training_data.x_raw, alpha=0.5)
plt.show()
```

## 12. What to change first

After running an example successfully, the safest first modifications are:

1. change `threshold` or add it to `solver_bounds`;
2. change weak-form parameters such as `window_length` and `stride`;
3. reduce `n_gen` and `pop_size` for quick debugging;
4. increase `n_gen` and `pop_size` for longer optimization runs;
5. modify the candidate library by adding or removing terms.

For more background on the package structure and terminology, see `docs/concepts.md`.
