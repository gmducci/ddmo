# Figure 1 — Parameterized-library recovery

This campaign compares an oracle Brusselator library with an adaptive library
that must recover the exponents in the interaction term. It is the only figure
campaign whose runner and postprocessor are currently complete.

## Plan jobs

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_1/config.toml
```

## Run

Smoke test:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_1.run \
  benchmarks/figures/figure_1/config.toml --job-id 0 --smoke
```

Production job:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_1.run \
  benchmarks/figures/figure_1/config.toml --job-id 0
```

## Postprocess


```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_1.postprocess
```

Generated files:

- `results/aggregated/feature_recovery_runs.csv`: one selected model per row;
- `results/aggregated/feature_recovery_quality.json`: completeness and
  provenance checks;
- `results/figures/feature_recovery_summary.{png,pdf}`: parameter recovery and
  held-out rollout accuracy;
- `results/figures/feature_recovery_robustness.{png,pdf}`: failed-candidate
  fraction and runtime;
- `results/figures/feature_recovery_pareto.{png,pdf}`: available Pareto fronts
  with selected models circled.

Smoke runs, failed outputs, and results without a finite test NRMSE are excluded
automatically. The quality report is the authoritative go/no-go check:
`publication_ready` becomes true only when every configured method and seed is
present and all included runs report a clean Git worktree.

The PNG files are convenient previews. Use the vector PDF files in the paper.
