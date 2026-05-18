# Goodwin oscillator with the weak formulation

This tutorial explains the example script:

```text
examples/tutorial_goodwin_weak.py
```

The example demonstrates a complete DDMO workflow for identifying a dynamical model from noisy Goodwin oscillator data using the weak formulation.

The weak formulation is useful when pointwise derivative estimates are unreliable. Instead of fitting the model directly from numerical derivatives, DDMO constructs an integrated regression problem using test functions over local windows.

## What this example demonstrates

This example shows how to:

- load the built-in Goodwin oscillator benchmark data;
- define a candidate library with polynomial, Hill, and exponential terms;
- configure a weak-form identification problem;
- optimize library and weak-form parameters;
- fit a sparse model with the PySINDy backend;
- compare the identified trajectory with the observed noisy data.

## Running the example

From the repository root, run:

```bash
PYTHONPATH=src python examples/tutorial_goodwin_weak.py
```

If the package has been installed in editable mode, run:

```bash
python examples/tutorial_goodwin_weak.py
```

The example prints the selected parameter vectors and displays a plot comparing the identified trajectory with the noisy observed data.

## Workflow overview

The script follows this structure:

```text
load benchmark data
    ↓
build candidate library
    ↓
construct DDMO data object
    ↓
configure optimizer
    ↓
configure sparse-regression solver
    ↓
configure weak formulation
    ↓
run identify(...)
    ↓
plot identified and observed trajectories
```

This structure mirrors the standard DDMO workflow: data, library, formulation, solver, and optimizer are configured separately and then passed to `identify`.

## Loading the benchmark data

The example loads the built-in Goodwin benchmark arrays from `BENCHMARK_FOLDER`:

```python
x_0_train = np.load(BENCHMARK_FOLDER / "x_0.npy")
x_noisy = np.load(BENCHMARK_FOLDER / "x_train_noise.npy")
time_array = np.load(BENCHMARK_FOLDER / "time.npy")
```

The helper function returns a dictionary containing:

```python
{
    "x_0": [x_0_train],
    "x_train": x_noisy[:, :],
    "x_raw": x_noisy[:, :],
    "time_array": time_array[:, :],
}
```

Here, `x_train` is the trajectory used for identification, while `x_raw` is kept for comparison and plotting.

## Building the candidate library

The candidate library defines the mathematical terms that may appear in the discovered model:

```python
library = (
    LibraryBuilder()
    .add_polynomial(max_degree=2, include_interactions=True)
    .add_hill(exponent_param="gamma_1")
    .add_exponential(rate_param="gamma_2")
)
```

This library combines three types of candidate terms:

- polynomial terms up to degree 2;
- Hill-type terms controlled by `gamma_1`;
- exponential terms controlled by `gamma_2`.

The parameters `gamma_1` and `gamma_2` are not fixed in the script. They are later optimized through `OptimizerSettings`.

Conceptually, the library answers the question:

```text
Which mathematical terms is the model allowed to use?
```

If a term is not included in the library, the sparse-regression solver cannot select it.

## Constructing the data object

The DDMO data object is created with:

```python
training_data = DataConstructor(
    x_train=data["x_train"],
    time_array=data["time_array"],
    library_object=library.build,
    x_0=data["x_0"],
    x_raw=data["x_raw"],
)
```

This object collects the trajectory, time array, candidate library, initial condition, and raw data needed by the identification pipeline.

In this weak-form example, derivative data are not passed explicitly. The weak formulation constructs the regression problem using integrated quantities rather than relying directly on pointwise derivative estimates.

## Configuring the optimizer

The optimizer searches over both library parameters and weak-form parameters.

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
```

The library bounds control parameters inside the candidate library. The formulation bounds control parameters of the weak formulation.

The optimizer settings are:

```python
optimizer_settings = OptimizerSettings(
    library_bounds=library_bounds,
    formulation_bounds=formulation_bounds,
    n_obj=2,
    n_gen=10,
    algorithm="NSGA3",
    pop_size=50,
    seed=42,
)
```

The important fields are:

- `library_bounds`: parameters of the candidate library to optimize;
- `formulation_bounds`: weak-form parameters to optimize;
- `n_obj`: number of objectives;
- `n_gen`: number of optimizer generations;
- `algorithm`: optimization algorithm;
- `pop_size`: population size;
- `seed`: random seed for reproducibility.

The values in this example are intended to make the workflow concrete. For more demanding experiments, `n_gen` and `pop_size` may need to be increased.

## Configuring the solver

The sparse-regression solver is configured with `SolverSettings`:

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

In this configuration:

- `solver_name="pysindy"` selects the PySINDy backend;
- `regularization="STLSQ"` selects the sparse-regression optimizer;
- `regularization_parameters` are forwarded to the selected optimizer;
- `scale_features=True` applies DDMO-level feature scaling before fitting.

The distinction between `scale_features` and `normalize_columns` is important.

```text
scale_features=True
    DDMO scales the regression problem before fitting and rescales the coefficients afterward.

normalize_columns=False
    STLSQ does not perform its own additional column normalization internally.
```

This allows DDMO to control feature scaling explicitly while avoiding an additional optimizer-level normalization step.

## Configuring the weak formulation

The initial weak-form settings are:

```python
formulation_settings = FormulationSettings(
    formulation="weak",
    test_function="PolynomialWindow",
    degree=2,
    window_length=65,
    stride=50,
)
```

These settings define the initial weak formulation before optimization.

The main parameters are:

- `formulation="weak"`: use the weak formulation;
- `test_function="PolynomialWindow"`: use polynomial window test functions;
- `degree`: polynomial degree of the test function;
- `window_length`: number of time points in each weak-form window;
- `stride`: spacing between consecutive windows.

In this example, `degree`, `window_length`, and `stride` are also included in `formulation_bounds`, so the optimizer can search over them.

## Running identification

The full identification call is:

```python
result = identify(
    data_instance=training_data,
    optimizer_settings=optimizer_settings,
    solver_settings=solver_settings,
    formulation_settings=formulation_settings,
)
```

This call combines the configured data, optimizer, solver, and weak formulation.

The returned `result` contains the solutions found by the optimization process and the selected parameter vectors:

```python
print("Selected parameter vectors / alphas:")
print(result.alphas)
```

The exact structure of the result object depends on the optimization output, but the example uses the first identified solution for plotting:

```python
result.solutions[0]
```

## Plotting the result

The example compares the first identified trajectory with the observed noisy trajectory:

```python
ax.plot(result.solutions[0], label="identified trajectory")
ax.plot(observed, alpha=0.5, linestyle="--", label="observed noisy data")
```

This plot is intended as a quick qualitative check. It does not replace a full numerical evaluation of model quality.

A good follow-up improvement is to add a reporting utility that prints the discovered symbolic equations, for example:

```text
dx0/dt = ...
dx1/dt = ...
dx2/dt = ...
```

That functionality should eventually live in a reusable package-level utility rather than only inside this example.

## How to modify this example

### Make the optimization faster

Reduce the number of generations or the population size:

```python
n_gen=3
pop_size=20
```

This is useful for debugging, but may produce lower-quality solutions.

### Make the optimization more thorough

Increase the number of generations or the population size:

```python
n_gen=50
pop_size=100
```

This may improve the search but will increase runtime.

### Change the weak-form search space

Modify `formulation_bounds`:

```python
formulation_bounds = {
    "degree": (2, 5),
    "stride": (25, 150),
    "window_length": (101, 501),
}
```

This changes the range of weak-form configurations explored by the optimizer.

### Change sparse-regression strength

Modify the STLSQ threshold:

```python
"threshold": 0.05
```

A smaller threshold usually allows more terms to remain active. A larger threshold usually encourages sparser models.

### Change the candidate library

Modify the library builder:

```python
library = (
    LibraryBuilder()
    .add_polynomial(max_degree=3, include_interactions=True)
)
```

This changes the terms available to the solver. A richer library can represent more dynamics, but it can also make the regression problem harder and less interpretable.

## Interpreting this example

The example should be understood as a tutorial workflow rather than a final benchmark.

It demonstrates how DDMO combines:

```text
benchmark trajectory data
+ candidate library
+ weak formulation
+ sparse-regression solver
+ multi-objective optimization
```

to produce an identified dynamical model.

The most important design point is that each part of the workflow is configured explicitly. This makes the example easier to modify, document, and eventually test.