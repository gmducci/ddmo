# Paper benchmarks

**Figures to use in the paper are in [figures/](figures/).**
No manuscript, legacy experiments, campaign snapshots, or exploratory plots live here.

```text
benchmarks/
  README.md
  run.py                 # production campaigns
  plot.py                # regenerate or verify the paper figures
  diagnostics.py         # shared supplementary trajectory diagnostics
  noise/                 # Figure 1 + Figure S1: config, runner, plotting
  joint/                 # Figure 2: config, initial conditions, runner, plotting
  common/                # shared execution, data checks, and styling
  data/                  # result ZIPs, trajectory NPZs, provenance manifest
  figures/               # six selected figures, PDF + PNG, and hash manifest
```

| Figure | Content | Files |
|---|---|---|
| 1 | Noise robustness: Strong, Weak optimized, Weak optimized + SG | [PDF](figures/figure_1_noise_robustness.pdf), [PNG](figures/figure_1_noise_robustness.png) |
| 2 | Selection frequency, conditional median coefficients, ground truth | [PDF](figures/figure_2_library_recovery.pdf), [PNG](figures/figure_2_library_recovery.png) |
| S1 | All library coefficients for the three noise-benchmark methods | [PDF](figures/figure_s1_full_coefficient_recovery.pdf), [PNG](figures/figure_s1_full_coefficient_recovery.png) |
| S2 | Figure 1 diagnostics: all noise/seed pairs, three methods, both test trajectories (25 pages) | [PDF](figures/figure_s2_noise_diagnostics.pdf), [Preview](figures/figure_s2_noise_diagnostics.png) |
| S3 | Figure 2 diagnostics: all ten runs, including training, validation, and test trajectories (10 pages) | [PDF](figures/figure_s3_joint_diagnostics.pdf), [Preview](figures/figure_s3_joint_diagnostics.png) |
| S4 | Figure 2 overview: representative test rollout with Hill/exponential shapes on the right | [PDF](figures/figure_s4_joint_library_shapes.pdf), [PNG](figures/figure_s4_joint_library_shapes.png) |

Each S2/S3 diagnostic page also includes the three-objective Pareto projection and
selected coefficients. S2 excludes fixed weak and labels failed rollouts.
Diagnostic PNGs preview the first page; the PDFs contain all pages.

S4 restores the former Figure 2 trajectory/library layout for the current ten-run
campaign. The example is repetition 4, test trajectory 1: the run closest to the
median validation NRMSE (repetition order breaks ties). Reported test NRMSE covers
both test trajectories. The right panel shows unweighted library functions, not
coefficient-weighted contributions to the ODE: `1/(1+s**gamma_1)` and
`exp(-gamma_2*s)`. Faint curves show individual active-family parameters; bold
curves use their conditional median. Hill is active in 6/10 runs and exponential
in 4/10. Parameters from inactive families are excluded.

## Regenerate figures

Use the project environment (Python 3.11+, tested on 3.12) from the repository root:

```bash
python -m benchmarks.plot --check
python -m benchmarks.plot
```

The first command checks the selected files, numerical inputs, and run provenance.
The second regenerates all six figures from numerical results into
`.local/benchmark_plots/`. It does not run optimization or overwrite the selected
figures. Use `noise` or `joint` to render only one experiment:

```bash
python -m benchmarks.plot joint --output-directory .local/joint_preview
python -m benchmarks.plot --diagnostics
python -m benchmarks.plot joint --library-overview
```

The two result ZIPs are versioned and total less than 1 MB. They contain the exact
production JSON results and original configuration; the joint bundle also includes
the original initial-condition design. Individual file hashes and original source
commits are in `data/manifest.json`. Archived source paths inside result metadata
are provenance only: figure regeneration does not read those paths or the local archive.
Two deduplicated trajectory NPZs (about 11 MB total) supply the exact saved arrays
for S2-S4. All six figures regenerate without the local archive. Page-to-run
assignments are recorded in `figures/manifest.json`; regeneration also writes
page-index JSON files alongside the generated diagnostics in `.local/`.

## Run new production campaigns

From a clean, committed source tree:

```bash
python -m benchmarks.run noise --workers 5
python -m benchmarks.run joint --workers 5
```

Each command runs/resumes its full campaign and then renders its figures. New
raw results, logs, and plots go to `.local/benchmark_runs/<experiment>/`; the
bundled publication data remain unchanged. A cached job is reused only when its
commit, config hash, artifact, and evaluation budget match.

| Experiment | Design | Production budget |
|---|---|---|
| Noise | Three methods, five noise levels, five seeds; 3/1/2 training/validation/test split | 75 runs × 2,400 evaluations |
| Joint | Ten repetitions at 10% noise; 2/1/2 split; Latin-hypercube initial conditions | 10 runs × 2,400 evaluations |

Both minimize trajectory RMSE, absolute DW deviation from two, and active
coefficient count. Joint optimization searches the Hill/exponential library,
weak-formulation settings, and STLSQ settings. Configurations are in
`noise/config.toml` and `joint/config.toml`.

The saved noise data are the 75 displayed runs from the original 100-run campaign;
fixed weak is excluded. New noise runs execute only the three retained methods.
Two saved Strong test rollouts failed at 20% noise: that test-error median uses
three successful runs; coefficient statistics include all five fitted models.
SG weak objectives use smoothed observations. Validation/test NRMSE uses clean
synthetic references. These scientific choices have not changed during cleanup.

Figure 2 reports medians **when selected**; blank cells mean never selected.
Coefficient colors are clipped to the ground-truth range, but annotations give
the actual values. Hill/exponential parameters vary by run, so the coefficient
summary is not a single recovered dynamical model.

## Local archive

The complete previous benchmark folder, including raw trajectories, old figures,
diagnostics, and frozen source snapshots, is preserved under
`.local/benchmark_archive/20260922_162348/`. It is ignored by Git and is not a
dependency of the active benchmark package. No production optimization was rerun
and no previously selected artwork was changed by this cleanup.
