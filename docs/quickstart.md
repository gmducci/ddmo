# Quickstart

This page summarizes the typical `ddmo` workflow.

## Basic workflow

A standard identification workflow consists of:

1. preparing trajectory data;
2. defining a candidate function library;
3. configuring the formulation;
4. configuring the sparse regression solver;
5. configuring the optimizer;
6. running `identify`.

## Run an existing example

From the repository root:

```bash
python examples/ex_goodwin_weak.py
```

or:

```bash
python examples/ex_goodwin_strong.py
```

## Main public API

The most common user-facing imports are available directly from `ddmo`:

```python
from ddmo import (
    DataConstructor,
    FormulationSettings,
    OptimizerSettings,
    SolverSettings,
    identify,
)
```

Additional model definitions and library builders are available from their submodules:

```python
from ddmo.library_functions import LibraryBuilder
from ddmo.equations.goodwin_eqs import GoodwinModel
```

## Output

The `identify` function returns an identification result containing optimization outputs and objective-space values.

Generated figures, serialized results, local analysis folders, and large numerical outputs should be written to ignored result directories rather than committed to Git.

