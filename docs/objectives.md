# Optimization objectives

`ddmo` represents objectives as an ordered mapping from readable names to
callables. Each callable receives one `ObjectiveContext` containing the input
data and complete solver result.

Without an explicit mapping, `identify` uses trajectory mean squared error and
the worst absolute Durbin-Watson deviation from two:

```python
result = identify(
    data_instance=data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
)
```

## Optional sparsity objective

Active-coefficient count is available but is not enabled by default:

```python
from ddmo.optimizer import make_active_coefficient_count, trajectory_mse

objectives = {
    "trajectory_mse": trajectory_mse,
    "active_coefficients": make_active_coefficient_count(tolerance=1e-6),
}

result = identify(
    data_instance=data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
    objective_functions=objectives,
)
```

The number of objectives is derived from this mapping. Users do not need to
keep a separate `n_obj` value synchronized with it.

## Custom objectives

A custom objective is an ordinary callable returning one scalar to minimize:

```python
import numpy as np


def coefficient_l1_norm(context):
    return float(np.sum(np.abs(context.coefficients)))


objectives = {
    "coefficient_l1_norm": coefficient_l1_norm,
}
```

The context exposes:

- `context.data`;
- `context.solver_result`;
- `context.solution`;
- `context.coefficients`.

Objective names define the column order in the returned Pareto objective
matrix, so use stable descriptive names in reproducible studies.
