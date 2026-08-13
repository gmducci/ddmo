# Figure 2 — Noise and formulation robustness

This campaign compares strong, fixed weak, optimized weak, and SG-preprocessed
optimized weak formulations on Goodwin trajectories at 0%, 5%, 10%, 15%, and
20% noise. The SG-preprocessed weak arm uses the same 21-point, degree-3 filter
as the strong baseline, then applies the weak integral formulation without
pointwise differentiation.

## Current status

- Experiment configuration: ready.
- Job planning: ready through `benchmarks.common.plan_runs`.
- Scientific runner: ready; all three method paths passed smoke verification.
- Trajectory artifacts: saved for every selected model.
- Postprocessor: ready; generates the three-panel noise-robustness figure.

The postprocessor plots held-out test NRMSE, coefficient error, and invalid-
candidate fraction against noise level:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_2.postprocess
```

Inspect the planned 100-job matrix with:

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_2/config.toml
```

Smoke-test one job by zero-based index:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_2.run \
  --job-id 0 --smoke
```

For automation, prefer the stable semantic job identifiers printed by the
planner, for example:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_2.run \
  --job-id weak_noise-weak_optimized-noise020-seed055
```

Zero-based indices remain available for interactive use. Omit `--smoke` only
after all three method paths have passed.

After committing all benchmark code, run or resume the strict publication
campaign with:

```bash
python -m benchmarks.figures.figure_2.run_campaign --workers 6
```

The supervisor refuses a dirty worktree and reuses a cached job only when its
clean Git commit, config hash, evaluation budget, JSON, and NPZ artifact all
match the current campaign.
