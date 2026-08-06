# Publication benchmarks

This directory contains the frozen experiment definitions for the minimal
software-paper validation. It is intentionally separate from `examples/`:

- `examples/` teaches the API;
- `benchmarks/` produces reproducible evidence for the paper;
- generated benchmark outputs are ignored by Git.

## Experiment set

1. `feature_recovery.toml` isolates parameterized-library recovery on the
   Brusselator.
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

The scientific runner will consume this manifest only after the core objective
and weak-window semantics are corrected and tested. Final configurations must
be committed before production runs; configuration changes after inspecting
final outcomes require a new experiment version.

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
