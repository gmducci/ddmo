# Figure 2 — Noise and formulation robustness

This campaign will compare strong, fixed weak, and optimized weak formulations
on Goodwin trajectories at 0%, 5%, and 10% noise.

## Current status

- Experiment configuration: ready.
- Job planning: ready through `benchmarks.common.plan_runs`.
- Scientific runner: not implemented.
- Postprocessor: not implemented.

The final postprocessor will plot held-out test NRMSE, coefficient or support
error, and invalid-candidate fraction against noise level. Do not launch this
campaign until its runner passes smoke tests for all three methods.

Inspect the planned 45-job matrix with:

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_2/config.toml
```
