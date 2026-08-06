# Goodwin weak-form tutorial

This tutorial explains `examples/quickstart/goodwin_weak.py`.

The script identifies a Goodwin oscillator model from noisy benchmark data
using the weak formulation. The weak formulation is useful when numerical
derivatives are unreliable because it builds the regression problem from
windowed integral quantities.

Run it from the repository root:

```bash
PYTHONPATH=src python examples/quickstart/goodwin_weak.py
```

After editable installation:

```bash
python examples/quickstart/goodwin_weak.py
```

## Workflow

The script follows the same order as the package workflow:

```text
load data
build candidate library
construct DataConstructor
configure OptimizerSettings
configure SolverSettings
configure FormulationSettings
call identify(...)
print parameters and plot the first solution
```

## 1. Load data

```python
x_0_train = np.load(BENCHMARK_FOLDER / "x_0.npy")
x_noisy = np.load(BENCHMARK_FOLDER / "x_train_noise.npy")
time_array = np.load(BENCHMARK_FOLDER / "time.npy")
```

`x_noisy` is used both as training data and as the raw trajectory shown in the
plot.

## 2. Build the candidate library

```python
library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2, include_interactions=True)
    .add_hill(exponent_param="gamma_1")
    .add_exponential(rate_param="gamma_2")
)
```

The library defines the terms the solver is allowed to select. `gamma_1` and
`gamma_2` are not fixed here; they are optimized later.

## 3. Construct the data object

```python
training_data = DataConstructor(
    x_train=data["x_train"],
    time_array=data["time_array"],
    x_0=data["x_0"],
    x_raw=data["x_raw"],
)
```

No derivative array is passed. That is the main practical difference from a
strong-form workflow.

## 4. Configure optimization

```python
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
    seed=42,
)
```

`library_bounds` tune parameters inside the candidate library.
`formulation_bounds` tune the weak-form test functions.

For quick debugging, reduce `n_gen` and `pop_size`. For a more serious search,
increase them.

## 5. Configure the solver

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

The solver fits sparse coefficients after the weak formulation has built the
regression matrices.

## 6. Configure the weak formulation

```python
formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=65,
    stride=50,
)
```

These are the starting values. Because `degree`, `window_length`, and `stride`
also appear in `formulation_bounds`, the optimizer can change them.

## 7. Run identification

```python
result = identify(
    data_instance=training_data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
    library_object=library.build,
)
```

The tutorial prints the selected optimizer parameters with names and plots
`result.solutions[0]` against the noisy data.

## What to change first

- Faster run: decrease `n_gen` and `pop_size`.
- Cleaner or noisier data: replace `x_train` and `x_raw`.
- Different weak-form search: change `degree`, `stride`, and `window_length`.
- Different sparse model: change `threshold` or add it to `solver_bounds`.
- Different candidate terms: edit the `LibraryBuilder` chain.

For the smallest package overview, see `docs/tutorials/minimal_workflow.md`.
