# Publication benchmarks

This directory contains the frozen experiment definitions for the minimal
software-paper validation. It is intentionally separate from `examples/`:

- `examples/` teaches the API;
- `benchmarks/` produces reproducible evidence for the paper;
- generated benchmark outputs are ignored by Git.

## Experiment set

1. `feature_recovery.toml` isolates parameterized-library recovery on the
   Brusselator by comparing the adaptive library with an oracle library whose
   exponents are fixed to their known true values.
2. `weak_noise.toml` compares strong, fixed weak, and optimized weak
   formulations on noisy Goodwin trajectories.
3. `joint_goodwin.toml` exercises the complete joint library, formulation, and
   solver search at 10% noise.

Every experiment uses three training trajectories, one validation trajectory
for Pareto-model selection, and two untouched test trajectories.

## Run planning

Inspect the exact job matrix without running optimization:

```bash
python benchmarks/plan_runs.py benchmarks/configs/feature_recovery.toml
```

Write a CSV job manifest:

```bash
python benchmarks/plan_runs.py benchmarks/configs/weak_noise.toml \
  --output benchmarks/results/weak_noise_jobs.csv
```

Run the first feature-recovery job through the complete pipeline with a small
smoke budget:

```bash
python benchmarks/run_job.py benchmarks/configs/feature_recovery.toml \
  --job-id 0 --smoke
```

Smoke mode uses one training, validation, and test trajectory, a two-time-unit
horizon, and a 12-by-3 optimizer budget. It exercises the production code path
but does not produce scientific evidence. Its output is written to
`benchmarks/results/raw/feature_recovery/feature_recovery-001-smoke.json`.

After the smoke output has been inspected, run the same job at its committed
production settings by omitting `--smoke`. Existing outputs are protected from
accidental replacement; `--force` must be supplied explicitly to overwrite one.

The runner currently supports `feature_recovery`. The weak-noise and joint
Goodwin runners will be added only after this first pipeline is verified. Final
configurations must be committed before production runs; configuration changes
after inspecting final outcomes require a new experiment version.

## Output contract

Each completed job must record:

- Git commit and dirty-state flag;
- package and dependency versions;
- experiment, method, noise level, and seed;
- optimizer budget and actual candidate evaluations;
- full Pareto objective values and decision variables;
- selected coefficients and symbolic equations;
- validation and test metrics;
- failed-candidate count and wall time.

Raw outputs go under `benchmarks/results/raw/`. Aggregated tables and final
figures go under `benchmarks/results/aggregated/` and
`benchmarks/results/figures/`. These directories are ignored because archival
results should be deposited separately from source code.

## Postprocessing previews

Generate the current feature-recovery tables, quality report, and figure
previews at any point during the campaign:

```bash
MPLBACKEND=Agg python -m benchmarks.postprocessing.feature_recovery
```

The postprocessor excludes smoke and failed files and visibly labels figures
as previews until every configured seed is present with clean Git provenance.
See `postprocessing/README.md` for the complete output list.
