# Concepts

`ddmo` is a Python package for data-driven model discovery of differential equations.

The package is designed around a modular identification workflow. Starting from trajectory data, `ddmo` builds a regression problem, evaluates candidate model terms, fits sparse coefficients, and optionally optimizes over configuration parameters of the identification pipeline.

The main goal is to make model discovery workflows configurable, reproducible, and extensible.

## High-level workflow

A typical `ddmo` workflow follows this structure:

```text
raw trajectory data
    ↓
data construction
    ↓
candidate library construction
    ↓
strong or weak formulation
    ↓
sparse-regression solver
    ↓
identified model
    ↓
optimization and diagnostics
```

Each stage has a specific responsibility:

- the data layer stores and validates the observed trajectories;
- the library layer defines the candidate terms that may appear in the model;
- the formulation layer defines how the regression problem is constructed;
- the solver layer estimates sparse model coefficients;
- the optimization layer searches over formulation, library, and solver parameters.

This separation is intentional. It allows users to modify one part of the identification pipeline without rewriting the others.

## Model discovery problem

``ddmo`` addresses problems of the following form:

Given observed data from a dynamical system, identify a compact differential-equation model that explains the observed dynamics.

For example, given measurements of a state vector over time,

```text
x(t) = [x_1(t), x_2(t), ..., x_n(t)]
```

the objective is to identify equations such as:

```text
dx/dt = f(x)
```

or, more generally, equations involving candidate nonlinear functions, derivatives, and other terms.

The discovered model is represented as a sparse combination of candidate functions. Sparsity is important because many scientific models depend only on a small number of active terms, even if the candidate library is large.

## Data

The data object represents the observed trajectories used for model identification.

In `ddmo`, data are constructed before running the identification pipeline. The data layer is responsible for storing quantities such as:

- observed state variables;
- time information;
- variable names or metadata;
- trajectory structure;
- information required by the formulation and solver stages.

The data object is shared by the rest of the pipeline. Candidate libraries, formulations, solvers, and optimizers all operate on the same underlying representation of the observed system.

Conceptually, the data layer answers the question:

**What observations are available for model discovery?**

## Candidate libraries

A candidate library defines the terms that are allowed to appear in the discovered model.

For example, a polynomial library may contain terms such as:

```text
1, x, y, x^2, xy, y^2
```

A richer library may include derivatives, trigonometric functions, custom functions, or terms specific to a particular physical problem.

The library defines the search space of possible equations. If a term is not included in the library, the solver cannot select it.

Conceptually, the library layer answers the question:

**Which mathematical terms may appear in the model?**

A good candidate library should be expressive enough to contain the true dynamics, but not so large that the regression problem becomes unnecessarily unstable or difficult to interpret.

## Formulations

A formulation defines how `ddmo` transforms trajectory data and candidate library terms into a regression problem.

The formulation is one of the most important parts of the model discovery pipeline because it determines what the solver actually sees.

`ddmo` supports different formulation strategies, including strong and weak formulations.

### Strong formulation

The strong formulation uses pointwise information about the dynamics.

In a typical strong-form approach, one estimates time derivatives from data and solves a regression problem of the form:

**time derivatives ≈ candidate library × coefficients**

The strong formulation is usually simple and computationally efficient. It is often a natural first choice when the data are clean and derivative estimates are reliable.

However, strong formulations can be sensitive to noise. Numerical differentiation may amplify measurement noise, and this can degrade the quality of the identified model.

The strong formulation is often appropriate when:

- the data are relatively clean;
- the time resolution is sufficient;
- derivative estimates are stable;
- a simple and fast baseline is needed.

### Weak formulation

The weak formulation constructs the regression problem through integration against test functions.

Instead of relying directly on pointwise derivative estimates, the weak formulation works with integrated quantities. This can make the identification problem more robust when the data are noisy or when derivative estimates are unreliable.

Weak formulations usually introduce additional configuration choices, such as:

- the family of test functions;
- the window length;
- the stride between windows;
- the polynomial degree or shape parameters of the test functions.

Such additional configurations, are often parameters dependent. Such parameters fine tuning can still be handled by the `ddmo` optimization layer. 

The weak formulation is often appropriate when:

- the data are noisy;
- pointwise derivatives are unstable;
- robustness is more important than simplicity;
- the user wants to exploit integral information over local windows.

The additional parameters make weak formulations more flexible, but also introduce a larger configuration space. This is one reason why optimization over formulation parameters can be useful.

## Solvers

Once the formulation has constructed a regression problem, the solver estimates the model coefficients.

In sparse model discovery, the solver should usually favor compact models. This means that many candidate terms should receive zero or near-zero coefficients, leaving only the most relevant terms active.

In `ddmo`, solver behavior is configured through `SolverSettings`.

A typical solver configuration may look like this:

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

- `solver_name` selects the solver backend;
- `regularization` selects the sparse-regression optimizer used by that backend;
- `regularization_parameters` are forwarded to the selected optimizer;
- `scale_features` controls whether `ddmo` applies feature scaling before fitting.

For the PySINDy backend, `regularization="STLSQ"` selects the STLSQ optimizer. Parameters such as `threshold`, `alpha`, and `normalize_columns` are passed to the corresponding PySINDy optimizer.

Conceptually, the solver layer answers the question:

**Given a regression problem, which candidate terms should be active in the model?**

## Feature scaling

Feature scaling can have a significant effect on sparse regression.

Candidate library terms may have very different numerical scales. For example, a constant term, a linear term, and a high-degree polynomial term may differ by several orders of magnitude. Without scaling, the solver may favor or penalize terms partly because of their numerical magnitude rather than their explanatory value.

In `ddmo`, `scale_features=True` enables `ddmo`-level scaling of the regression problem before fitting. The fitted coefficients are then transformed back to the original scale.

This is separate from optimizer-specific normalization options such as `normalize_columns` in STLSQ.

For example, the following configuration means that `ddmo` applies its own feature scaling, while the STLSQ optimizer does not apply additional column normalization internally:

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

This distinction is important:

```text
scale_features
    `ddmo`-level preprocessing before sparse regression.

normalize_columns
    optimizer-level behavior inside the selected sparse-regression method.
```

## Optimization

`ddmo` can optimize over parameters of the identification pipeline.

Instead of fitting a single fixed model, the optimizer evaluates multiple candidate configurations. These configurations may differ in formulation parameters, library parameters, or solver parameters.

For example, the optimizer may search over:

- weak-form window lengths;
- test-function parameters;
- library complexity;
- sparse-regression thresholds;
- regularization strengths.

Conceptually, the optimization layer answers the question:

**Which configuration of the model discovery pipeline gives the best trade-off between accuracy, sparsity, and complexity?**

This is especially useful because model discovery often involves competing objectives. A very accurate model may be too complex, while a very sparse model may underfit the data.

`ddmo` is designed to support this trade-off explicitly.

## Configuration layers

`ddmo` separates configuration into different settings objects.

The exact structure may evolve, but the core idea is:

```text
data settings
    define the observed system;

formulation settings
    define how the regression problem is constructed;

library settings
    define the candidate model terms;

solver settings
    define how coefficients are estimated;

optimizer settings
    define which parameters are searched over.
```

This separation improves readability and reproducibility. A complete experiment can be described by its settings, and individual components can be modified without changing the entire pipeline.

## Outputs

The output of a `ddmo` workflow depends on the specific identification and optimization setup, but it generally includes information such as:

- the discovered model coefficients;
- the selected active terms;
- the fitted sparse model;
- objective values;
- optimization diagnostics;
- metadata about the selected configuration.

The most important output is not only a fitted model, but also the information needed to understand how that model was obtained.

For scientific machine learning workflows, this is important because the identified equation should be interpretable, reproducible, and comparable across different configurations.

## How the pieces fit together

A minimal `ddmo` workflow can be understood as follows:

```python
# 1. Construct or load data
data_instance = ...

# 2. Configure the formulation
formulation_settings = ...

# 3. Configure the candidate library
library = ...

# 4. Configure the sparse-regression solver
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

# 5. Run model identification
result = identify(
    data_instance=data_instance,
    optimizer_settings=optimizer_settings,
    formulation_settings=formulation_settings,
    solver_settings=solver_settings,
    library_object=library.build,
)
```

The precise code depends on the specific example, but the conceptual structure remains the same:

```text
data + library + formulation + solver → identified model
```

With optimization enabled, the workflow becomes:

```text
data + configurable pipeline + parameter bounds → optimized identified model
```

## Design principles

`ddmo` is organized around a few design principles.

### Modularity

Each stage of the model discovery workflow has a separate responsibility. This makes it easier to replace or extend individual components.

### Reproducibility

Experiments should be describable through explicit configuration objects. Important modeling choices should not be hidden inside scripts.

### Extensibility

The package should support new libraries, formulations, solvers, and optimization strategies without requiring major changes to the rest of the codebase.

### Interpretability

The final model should be understandable as an equation or sparse combination of candidate terms. This distinguishes model discovery from purely black-box prediction.

### Research-to-software transition

`ddmo` is intended to support scientific model discovery while following software-engineering practices that make the code usable by others: clear configuration, documented abstractions, examples, and tests.
