# Examples

Examples are small, maintained demonstrations of the public `ddmo` API. They
are not publication benchmarks and do not store generated results in this
directory.

## Quickstart

- `quickstart/goodwin_weak.py` demonstrates the standard weak-form workflow.
- `quickstart/brusselator_custom_library.py` demonstrates a parameterized
  custom feature library.

## Advanced

- `advanced/goodwin_generated_batch.py` generates multiple synthetic
  trajectories before identification.
- `advanced/goodwin_multiple_trajectories.py` demonstrates weak identification
  with multiple trajectories.
- `advanced/brusselator_multiple_trajectories.py` combines a custom library
  with multiple trajectories.
- `advanced/goodwin_strong.py` demonstrates the strong formulation with
  supplied derivative estimates.

Run an example from the repository root after installing the package:

```bash
python examples/quickstart/goodwin_weak.py
```

Reproducible paper experiments, fixed configurations, and result-generation
tools belong in `benchmarks/`.
