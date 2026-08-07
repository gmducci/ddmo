# Benchmark postprocessing

This package turns raw benchmark JSON files into tidy tables, quality checks,
and publication-style figures. It is designed to work while a campaign is
incomplete, but it labels incomplete or dirty-provenance results as previews.

Run from the repository root:

```bash
MPLBACKEND=Agg python -m benchmarks.postprocessing.feature_recovery
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
