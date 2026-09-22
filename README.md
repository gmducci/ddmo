# ddmo

`ddmo` is a Python package for data-driven model optimization and sparse identification of ordinary differential equation systems from data.

The package is designed around a modular workflow: construct candidate function libraries, choose a strong or weak formulation, configure a sparse regression solver, and optimize model-identification settings with multi-objective optimization.

## Features

- **Function-library construction** through a fluent `LibraryBuilder` API.
- **Strong and weak formulations** for sparse model identification.
- **Sparse-regression solvers** built around PySINDy and related optimization tools.
- **Multi-objective optimization** of formulation, solver, and library parameters.
- **Maintained examples** for Goodwin-type and Brusselator dynamical systems.
- **Reproducible publication benchmarks** kept separate from user tutorials.
- **Modular package structure** separating data, formulation, solver, optimization, and utilities.

## Installation

```bash
git clone https://github.com/FHI-theory/ddmo.git
cd ddmo
pip install -e .
```

For development, install optional development dependencies if configured in `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

## Quickstart

Run one of the example workflows from the repository root:

```bash
python examples/quickstart/goodwin_weak.py
```

or:

```bash
python examples/quickstart/brusselator_custom_library.py
```

The main public API can be imported directly from `ddmo`:

```python
from ddmo import (
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
```

## Documentation

Project documentation is available in [`docs/`](docs/):

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Optimization objectives](docs/objectives.md)
- [Development guide](docs/development.md)

Maintained demonstrations are indexed in [`examples/`](examples/README.md).
Frozen paper experiments and their output contract are documented in
[`benchmarks/`](benchmarks/README.md).

The selected, renumbered paper artwork is collected in
[publication figures](benchmarks/figures).

## Development

Run the test suite with:

```bash
python -m pytest
```

Generated outputs, logs, local analysis folders, checkpoints, and large numerical result files should not be committed to the repository.

## Contributing

Contributions are welcome. Prefer small, focused changes with tests where appropriate.

A typical workflow is:

```bash
git checkout -b feature/my-change
python -m pytest
git commit -m "Describe the change"
git push origin feature/my-change
```

Then open a merge request or pull request.
