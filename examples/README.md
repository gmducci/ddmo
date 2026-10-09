# Examples

Examples are small demonstrations of the public `ddmo` API. They do not store
generated results in this directory.

## Quickstart

- `quickstart/goodwin_weak.py` runs one Figure 1 design with corrected weights
  and compares it with the historical saved run.
- `quickstart/goodwin_joint.py` reproduces one Figure 2 joint-library run.
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

The full paper campaigns, fixed configurations, and saved results are in
`benchmarks/`.
