# Development Guide

## Development workflow

Use small, focused commits:

```bash
git status
python -m pytest
git add <files>
git commit -m "Short descriptive message"
git push origin main
```

Prefer one logical change per commit.

## Testing

Run the full test suite with:

```bash
python -m pytest
```

For targeted testing:

```bash
python -m pytest tests/unit
python -m pytest tests/test_example.py
```

Tests should be deterministic and should not depend on long wall-clock timeouts, local files, or generated result folders.

## Code organization

The core package lives under:

```text
src/ddmo/
```

Examples live under:

```text
examples/
```

Tests live under:

```text
tests/
```

Documentation lives under:

```text
docs/
```

Generated outputs should be kept out of version control.

## Public API

User-facing imports should be exposed from `ddmo.__init__` when they are stable and intended for external use.

Preferred:

```python
from ddmo import DataConstructor, OptimizerSettings, SolverSettings
```

Avoid requiring users to import from deep internal modules unless the object is intentionally advanced or experimental.

## Local artifacts

Do not commit:

- `.DS_Store`;
- virtual environments;
- logs;
- generated figures;
- generated `.npy`, `.pkl`, `.csv`, or checkpoint files;
- local result folders.

