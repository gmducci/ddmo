# ddmo

`ddmo` is a data-driven model optimization package tailored for identifying mathematical models of ordinary differential equations from data.

## Features

- **Differentiation Module:** Smoothing, noise addition, and derivative estimation through Kalman smoothing, wavelet methods, and Total Variation Regularization.
- **Library Functions Module:** Fluent API (`LibraryBuilder`) for composing function libraries with polynomials, Hill functions, exponentials, and custom parameterized functions.
- **Formulation Module:** Strong and weak formulations for sparse regression.
- **Optimizer Module:** Multi-objective optimization (pymoo) with sparse regression (PySINDy, scikit-learn).
- **Modular Architecture:** Clean separation of data, formulation, solver, and optimization concerns.

## Installation

```bash
git clone https://github.com/FHI-theory/ddmo.git
cd ddmo
pip install -e .
```

## Running Examples

```bash
python examples/ex_goodwin_strong.py  # Strong formulation example
python examples/ex_goodwin_weak.py    # Weak formulation example
```

---

## Roadmap: Future Improvements

### High Priority

| Area | Issue | Recommendation |
|------|-------|----------------|
| **Objective Functions** | Hardcoded in optimizer | Allow user-defined objectives via main API without modifying `src/ddmo/optimizer` |
| **SolverResult Dataclass** | Solver returns fragile tuples | Implement `SolverResult` dataclass for consistent return types |
| **Error Handling** | Silent failures in solver | Surface errors with informative messages instead of returning `None` |

### Architecture Improvements

| Area | Current State | Target State |
|------|---------------|--------------|
| **Named Objectives** | List-based `[mse, dw]` | Dict-based `{"mse": fn, "dw": fn}` for generic handling |
| **Formulation Parameters** | Mixed in bounds dicts | Separate `library_bounds` and `formulation_bounds` (✅ done) |
| **Feature Matrix Builder** | Duplicated across solvers | Centralized `FeatureMatrixBuilder` class (✅ done) |
| **Registry Pattern** | Manual if/else chains | Use registries for solvers, formulations, test functions |

### Robustness Enhancements

- [ ] **Condition number diagnostics**: Log `np.linalg.cond(Θ)` before regression
- [ ] **Alternative optimizers**: SR3 is more robust than STLSQ for ill-conditioned problems
- [ ] **Adaptive thresholding**: Auto-tune threshold based on coefficient distribution
- [ ] **Cross-validation**: Add CV-based regularization parameter selection

### Testing & Quality

- [ ] **Unit test coverage**: Expand tests for `FeatureMatrixBuilder`, formulations, solvers
- [ ] **Integration tests**: End-to-end tests with known benchmark systems
- [ ] **Deterministic seeding**: Ensure reproducibility via `seed` parameter
- [ ] **Type hints**: Add comprehensive type annotations throughout

### Code Quality

- [ ] Remove unused imports and dead code paths
- [ ] Add docstrings to all public APIs
- [ ] Standardize naming conventions (snake_case everywhere)
- [ ] Add `py.typed` marker for static analysis

### Documentation

- [ ] API reference documentation
- [ ] Tutorial notebooks for common use cases
- [ ] Architecture diagram showing module relationships

---

## Contributing

Contributions are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourFeature`)
3. Commit your Changes (`git commit -m 'Add some feature'`)
4. Push to the Branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

