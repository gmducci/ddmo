# Benchmark execution workflow

This document records the pipeline used to plan, execute, validate, and
postprocess the publication benchmark jobs. It is intended both as an operator
guide for collaborators and as an audit trail for reviewers.

All commands below are run from the repository root. The frozen scientific
settings for each experiment live in `benchmarks/figures/figure_<N>/config.toml`.
Generated results are written below `benchmarks/results/` and are intentionally
ignored by Git; archived result bundles should therefore be distributed
separately from the source repository.

## Pipeline at a glance

```text
frozen TOML configuration
          |
          v
validate configuration and enumerate deterministic job matrix
          |  benchmarks.common.plan_runs
          v
check clean Git commit and audit cached results (Figures 2 and 3)
          |  benchmarks.common.run_campaign
          v
dispatch independent jobs to the figure-specific runner
          |  benchmarks.figures.figure_<N>.run
          v
generate seeded trajectories and freeze train/validation/test split
          |
          v
construct library, formulation, solver, optimizer, and objectives
          |
          v
call ddmo.identify() and obtain the Pareto set
          |
          v
select one model using validation rollout NRMSE
          |
          v
evaluate the selected model once on untouched test trajectories
          |
          v
write raw JSON metadata and, for Figures 2 and 3, NPZ trajectories
          |
          v
verify every expected job, then aggregate tables and render figures
          |  benchmarks.figures.figure_<N>.postprocess
          v
quality report + campaign manifest + publication figures
```

The validation split selects a model from the Pareto set. The test split does
not participate in optimization or selection; it is used only after a model
has been selected.

## 1. Prepare and test the environment

Install the package and development dependencies in an isolated Python
environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/benchmarks
```

For headless machines, set `MPLBACKEND=Agg` when invoking runners or
postprocessors. Use the same Python environment for planning, execution, and
postprocessing so that the dependency versions captured in each result match.

## 2. Inspect the frozen job matrix

Planning validates the TOML schema and deterministically expands methods,
noise levels, seeds, replicate blocks, and training-set sizes. It does not run
an optimization.

```bash
python -m benchmarks.common.plan_runs benchmarks/figures/figure_1/config.toml
python -m benchmarks.common.plan_runs benchmarks/figures/figure_2/config.toml
python -m benchmarks.common.plan_runs benchmarks/figures/figure_3/config.toml
```

The currently frozen matrices are:

| Figure | Experiment | Jobs | Candidate evaluations per job | Total candidate evaluations |
|---|---|---:|---:|---:|
| 1 | `feature_recovery` | 10 | 600 | 6,000 |
| 2 | `weak_noise_poly2` | 100 | 2,400 | 240,000 |
| 3 | `joint_goodwin_lhs_training_size` | 30 | 2,400 | 72,000 |

A reviewable CSV manifest can be created without running jobs:

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_3/config.toml \
  --output /tmp/figure_3_jobs.csv
```

Semantic job identifiers printed by the planner should be preferred in logs
and automation. Zero-based indices are supported for interactive runs.

## 3. Smoke-test one job

Smoke mode follows the production code path but reduces the trajectory length
and optimizer budget. Its outputs are marked `"mode": "smoke"` and are
excluded by every publication postprocessor.

Figure 1:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_1.run \
  benchmarks/figures/figure_1/config.toml \
  --job-id 0 --smoke \
  --output /tmp/figure_1_smoke.json
```

Figure 2:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_2.run \
  benchmarks/figures/figure_2/config.toml \
  --job-id weak_noise-weak_optimized-noise020-seed055 \
  --smoke \
  --output /tmp/figure_2_smoke.json \
  --artifact-output /tmp/figure_2_smoke.npz
```

Figure 3:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_3.run \
  benchmarks/figures/figure_3/config.toml \
  --job-id joint_goodwin_lhs_training_size-weak_joint_sg-train1-block01-noise010 \
  --smoke \
  --output /tmp/figure_3_smoke.json \
  --artifact-output /tmp/figure_3_smoke.npz
```

A successful smoke run confirms that data generation, library construction,
optimization, model selection, rollout, and serialization all complete. It is
not scientific evidence and must not be mixed with production results.

## 4. Freeze the source state

Before a production campaign, commit the configuration and all code used by
the run:

```bash
git status --short
git rev-parse HEAD
```

`git status --short` must be empty. The strict Figure 2 and Figure 3 campaign
supervisor enforces this requirement and refuses tracked or untracked changes.
Each job records the full commit hash, dirty-state flag, configuration SHA-256,
Python version, and relevant package versions.

Do not edit a configuration during a campaign. A changed configuration hash or
Git commit invalidates the cache for the strict campaign, even if a job has the
same name.

## 5. Run or resume a production campaign

### Figure 2

```bash
python -m benchmarks.figures.figure_2.run_campaign --workers 6
```

### Figure 3

```bash
python -m benchmarks.figures.figure_3.run_campaign --workers 5
```

Both commands call `benchmarks.common.run_campaign.run_campaign()`. The
supervisor performs the following operations:

1. Requires a clean Git commit.
2. Loads and validates the selected TOML configuration.
3. Rebuilds the complete deterministic job matrix.
4. Computes the configuration SHA-256.
5. Reuses a cached job only if its status, production mode, commit, config hash,
   job ID, evaluation budget, actual evaluation count, JSON, and NPZ artifact
   all match the current campaign.
6. Runs missing or stale jobs concurrently using the requested worker count.
7. Stores one log per job under the experiment's `raw/logs/` directory.
8. Retries each failed job once.
9. Refuses to postprocess if any expected job remains failed, missing, or stale.
10. Calls the figure-specific postprocessor and writes
    `campaign_manifest.json` only after successful verification.

The commands are resumable: rerunning the same command from the same clean
commit skips verified jobs. A stale result is replaced automatically. Reduce
`--workers` when memory or CPU contention is more important than throughput.

Figure 1 currently has no strict campaign supervisor. Its ten production jobs
are run individually:

```bash
for job in {0..9}; do
  MPLBACKEND=Agg python -m benchmarks.figures.figure_1.run \
    benchmarks/figures/figure_1/config.toml --job-id "$job"
done
```

Existing Figure 1 outputs are not overwritten unless `--force` is supplied.
Because this path does not enforce a clean commit or write a campaign manifest,
the generated quality report and per-job provenance must be inspected before
using the results in a publication.

## 6. What happens inside one job

The figure-specific `run.py` resolves the job and applies either production or
smoke settings. For the Goodwin campaigns, shared implementation lives in
`benchmarks/common/goodwin.py`.

1. **Generate data.** The configured ODE is integrated from frozen initial
   conditions. Noise is generated from the job's data seed.
2. **Freeze splits.** Training, validation, and test trajectories are separated
   before fitting. Figures 1 and 2 use three training, one validation, and two
   test trajectories in production. Figure 3 uses nested groups of one, two,
   or three training trajectories within each paired LHS block, plus one shared
   validation and two shared test trajectories.
3. **Preprocess.** The selected method determines whether states and/or
   derivatives receive Savitzky--Golay preprocessing and whether the strong or
   weak formulation is constructed.
4. **Build the search problem.** The code constructs the candidate library,
   formulation bounds, STLSQ solver bounds, NSGA-II settings, random seed, and
   configured objectives.
5. **Optimize.** `ddmo.identify()` evaluates the configured population for the
   configured number of generations and returns fitted models on a Pareto
   front.
6. **Reject invalid candidates.** Results without a simulatable model, a finite
   objective vector, or a valid coefficient matrix are excluded and counted.
7. **Select on validation data.** Valid Pareto models are rolled out on the
   validation split. The lowest validation rollout NRMSE wins; ties are broken
   by the active-coefficient count and then the first training objective.
8. **Evaluate on test data.** Only the selected model is rolled out against the
   untouched clean test reference.
9. **Serialize.** Metrics, decisions, Pareto solutions, coefficients, symbolic
   equations, budgets, failures, timing, and provenance are written to JSON.
   Figures 2 and 3 also write compressed trajectory arrays to NPZ.

## 7. Output contract

```text
benchmarks/results/
├── raw/<experiment>/
│   ├── <job-id>.json
│   ├── <job-id>.npz        # Figures 2 and 3
│   ├── logs/<job-id>.log   # strict campaigns
│   └── campaign_manifest.json
├── aggregated/
│   ├── *_runs.csv
│   ├── *_quality.json
│   └── figure-specific coefficient/selection tables
└── figures/
    ├── *.png               # preview/raster inspection
    ├── *.pdf               # manuscript/vector output
    └── *.svg               # selected reviewer workflow outputs
```

The raw JSON is the primary machine-readable record. It includes the complete
Pareto set and identifies the selected model. NPZ files contain clean, noisy,
preprocessed, and predicted trajectories needed to reconstruct the plots.

## 8. Postprocess and audit

The strict Figure 2 and Figure 3 campaign commands postprocess automatically.
To regenerate tables and plots from an existing raw result bundle without
rerunning optimization:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_1.postprocess
MPLBACKEND=Agg python -m benchmarks.figures.figure_2.postprocess
MPLBACKEND=Agg python -m benchmarks.figures.figure_3.postprocess
```

Review the corresponding quality reports:

```bash
python -m json.tool benchmarks/results/aggregated/feature_recovery_quality.json
python -m json.tool benchmarks/results/aggregated/figure_2_quality.json
python -m json.tool benchmarks/results/aggregated/figure_3_quality.json
```

For publication use, `publication_ready` must be `true`, the expected and
observed job sets must match, no dirty-provenance jobs may be present, and any
reported rollout failures must be investigated. For Figures 2 and 3, also
inspect the campaign manifest and confirm that its commit and configuration
hash correspond to the reviewed source archive.

PNG files are convenient visual previews. Use the vector PDF outputs in the
manuscript. Figure-specific assumptions and output names are documented in
each `benchmarks/figures/figure_<N>/README.md`.

## Reviewer-oriented Figure 3 supplements

Figure 3 provides two additional renderers that consume the existing result
bundle but do not run optimization:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_3.workflow
MPLBACKEND=Agg python -m benchmarks.figures.figure_3.si_validation
```

The first renders the computational pipeline as SVG, PDF, and PNG. The second
renders all selected validation reconstructions for supplementary inspection.

