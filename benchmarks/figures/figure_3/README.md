# Figure 3 — Complete joint Goodwin optimization

This campaign will demonstrate the complete library, weak-formulation, and
solver search at 10% noise, followed by rollout on untouched initial
conditions.

## Current status

- Experiment configuration: ready.
- Job planning: ready through `benchmarks.common.plan_runs`.
- Scientific runner: not implemented.
- Trajectory artifact writer: not implemented.
- Postprocessor: not implemented.

The runner must save the selected validation and test trajectories in addition
to scalar metrics so the final figure can show observed, reference, and model
rollouts. Do not launch this campaign until those artifacts are verified.

Inspect the planned five-job matrix with:

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_3/config.toml
```
