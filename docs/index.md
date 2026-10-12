# DDMO Documentation

`ddmo` is a Python package for data-driven model optimization and sparse identification of ordinary differential equation systems from data.

The package provides modular components for:

- constructing candidate function libraries;
- building strong and weak formulations;
- estimating sparse dynamical models;
- optimizing formulation and library parameters;
- evaluating identified models against trajectory data.

## Documentation sections

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Development guide](development.md)

## Repository layout

```text
src/ddmo/      Core package implementation
examples/      Minimal executable examples
tests/         Unit and integration tests
docs/          Project documentation
```

Generated outputs, logs, local experiment results, and analysis folders should not be committed to the repository.

