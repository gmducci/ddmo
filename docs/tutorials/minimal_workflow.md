# Minimal workflow

This tutorial shows the smallest useful `ddmo` workflow. The main idea is:

```text
library + data -> formulation -> solver -> optimizer -> identify
```

Each object has one job. Build them in that order, then pass them to
`identify`.

## 1. Imports

```python
from ddmo import (
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
from ddmo.library_functions import LibraryBuilder
```

## 2. Candidate library

The library defines which terms the discovered model may use.

```python
library = LibraryBuilder().add_polynomial(max_degree=2)
```

For a parameterized library, give parameter names that can later be optimized:

```python
library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2)
    .add_hill(exponent_param="gamma_1")
)
```

## 3. Data

`DataConstructor` stores only trajectory data, time points, and initial
conditions. The candidate library is passed later to `identify`.

```python
data = DataConstructor(
    x_train=x_train,
    time_array=time_array,
    x_0=[x_train[0]],
    x_raw=x_train,
)
```

For the strong formulation, also pass derivative data:

```python
data = DataConstructor(
    x_train=x_train,
    time_array=time_array,
    x_dot=x_dot,
    x_0=[x_train[0]],
    x_raw=x_train,
)
```

## 4. Formulation

The formulation decides how the regression problem is built from the data.

Use the weak formulation when you do not want to rely on pointwise derivative
estimates:

```python
formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=65,
    stride=50,
)
```

Use the strong formulation when reliable derivatives are available:

```python
formulation_settings = FormulationSettings(formulation="strong")
```

## 5. Solver

The solver estimates sparse coefficients for the candidate terms.

```python
solver_settings = SolverSettings(
    solver_name="pysindy",
    regularization="STLSQ",
    regularization_parameters={
        "threshold": 0.01,
        "alpha": 0.05,
    },
    scale_features=True,
)
```

## 6. Optimizer

The optimizer searches over selected library, formulation, and solver
parameters.

```python
optimizer_settings = OptimizerSettings(
    library_bounds={"gamma_1": (5.0, 10.0)},
    formulation_bounds={
        "window_length": (40, 140),
        "stride": (10, 50),
    },
    solver_bounds={"threshold": (0.001, 0.2)},
    n_gen=10,
    pop_size=40,
    algorithm="NSGA3",
)
```

Bounds use this format:

```python
{"parameter_name": (lower_bound, upper_bound)}
```

If a parameter should stay fixed, leave it out of the bounds and set it in the
corresponding settings object instead.

## 7. Run identification

```python
result = identify(
    data_instance=data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
)
```

The result contains the selected optimizer vectors, coefficient matrices, and
simulated solutions:

```python
print(result.alphas)
print(result.coeff_matrices)
```

## Complete examples

Run a tutorial script from the repository root:

```bash
PYTHONPATH=src python examples/quickstart/goodwin_weak.py
```

After installing the package in editable mode, `PYTHONPATH=src` is not needed:

```bash
python examples/quickstart/goodwin_weak.py
```
