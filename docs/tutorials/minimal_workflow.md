# Minimal Identification Workflow

This tutorial outlines the main objects used in a typical `ddmo` identification workflow.

The goal is to show how the package components fit together, rather than to reproduce a full scientific experiment.

## 1. Import the public API

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

## 2. Prepare trajectory data

A `DataConstructor` instance collects the data needed by the identification workflow:

```python
data = DataConstructor(
    x_train=x_train,
    time_array=time_array,
    x_dot=x_dot,
    library_object=library.build,
    x_0=[x_train[0]],
    x_raw=x_train,
)
```

The expected inputs are:

- `x_train`: training trajectory data;
- `time_array`: time points associated with the trajectory;
- `x_dot`: derivative estimates or ground-truth derivatives;
- `library_object`: candidate function library;
- `x_0`: initial condition used for simulation;
- `x_raw`: raw trajectory data, usually before preprocessing.

## 3. Define a candidate library

Candidate functions are constructed with `LibraryBuilder`:

```python
library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2)
)
```

The library determines which terms can appear in the identified dynamical system.

## 4. Configure the formulation

For a weak-formulation workflow:

```python
formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=11,
    stride=1,
)
```

The formulation controls how the regression problem is constructed from the data.

## 5. Configure the solver

```python
solver_settings = SolverSettings(
    solver_name="pysindy",
    regularization="STLSQ",
    regularization_parameters={"threshold": 0.01, "alpha": 0.05},
)
```

The solver controls the sparse-regression backend and regularization parameters.

## 6. Configure the optimizer

```python
optimizer_settings = OptimizerSettings(
    formulation_bounds={"window_length": (5, 20), "stride": (1.0, 2.0)},
    library_bounds={},
    n_obj=2,
    n_gen=2,
    pop_size=5,
    algorithm="NSGA2",
)
```

The optimizer searches over selected formulation and library parameters.

## 7. Run identification

```python
result = identify(
    data_instance=data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
)
```

The returned result contains optimization outputs, objective-space values, and model-identification results.

## Next steps

For executable examples, see:

```bash
python examples/ex_goodwin_weak.py
python examples/ex_goodwin_strong.py
```

