# Installation

## Clone the repository

```bash
git clone https://gitlab.mpcdf.mpg.de/fhi-theory/ddmo.git
cd ddmo
```

## Create or activate a Python environment

Using `pyenv`, for example:

```bash
pyenv virtualenv 3.12.7 ddmo-dev
pyenv local ddmo-dev
```

Alternatively, activate your preferred virtual environment.

## Install the package

Install the package in editable mode:

```bash
pip install -e .
```

For development, install optional development dependencies if configured in `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

## Verify the installation

```bash
python - <<'PY'
from ddmo import identify, DataConstructor, OptimizerSettings, SolverSettings, FormulationSettings

print("ddmo import OK")
PY
```

## Run tests

```bash
python -m pytest
```

