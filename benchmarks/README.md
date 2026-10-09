# Paper benchmarks

**Figures to use in the paper are in [figures/](figures/).**
No manuscript, legacy experiments, campaign snapshots, or exploratory plots live here.

The corrected-weight Figure 2 tutorial has a separately verified block 4
reference at [data/figure2_tutorial_block04.json](data/figure2_tutorial_block04.json).
The ten-run bundle and selected Figure 2/SI artwork below still belong to the
earlier campaign. They will be replaced together after the SI figure review.

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
  figures/               # eight selected figures, PDF + PNG, and hash manifest
```

| Figure | Content | Files |
|---|---|---|
| 1 | Noise robustness: Strong + SG, Weak optimized, Weak optimized + SG | [PDF](figures/figure_1_noise_robustness.pdf), [PNG](figures/figure_1_noise_robustness.png) |
| 2 | Selection frequency, conditional median coefficients, ground truth | [PDF](figures/figure_2_library_recovery.pdf), [PNG](figures/figure_2_library_recovery.png) |
| S1 | All library coefficients for the three noise-benchmark methods | [PDF](figures/figure_s1_full_coefficient_recovery.pdf), [PNG](figures/figure_s1_full_coefficient_recovery.png) |
| S2 | Figure 1 diagnostics: all noise/seed pairs, three methods, both test trajectories (25 pages) | [PDF](figures/figure_s2_noise_diagnostics.pdf), [Preview](figures/figure_s2_noise_diagnostics.png) |
| S3 | Figure 2 diagnostics: all ten runs, including training, validation, and test trajectories (10 pages) | [PDF](figures/figure_s3_joint_diagnostics.pdf), [Preview](figures/figure_s3_joint_diagnostics.png) |
| S4 | Figure 2 overview: representative test rollout with Hill/exponential shapes on the right | [PDF](figures/figure_s4_joint_library_shapes.pdf), [PNG](figures/figure_s4_joint_library_shapes.png) |
| S5 | True Hill and fitted exponential feedback as functions of time on both held-out test trajectories | [PDF](figures/figure_s5_exponential_feedback.pdf), [PNG](figures/figure_s5_exponential_feedback.png) |
| S6 | Goodwin equation mismatch at the true coefficients: strong numerical derivatives and weak time-window integrals | [PDF](figures/figure_s6_equation_mismatch.pdf), [PNG](figures/figure_s6_equation_mismatch.png), [Values](figures/figure_s6_equation_mismatch.json) |

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

S5 compares the **coefficient-weighted** terms in the $\dot x$ equation for all
four exponential-selected models (repetitions 2, 3, 5, and 10): the true
$k_1/[1+(z/K_1)^{\gamma}]$ evaluated on the clean reference $z(t)$ and each
fitted $c\exp(-\gamma_2\hat z(t))$ evaluated on the identified model's test
rollout $\hat z(t)$.
Each row is one repetition, with test trajectory 1 on the left and test
trajectory 2 on the right. Each panel shows the full $0$--$200$ interval with
one true and one fitted curve. The ordinate is linear from 0 to 1. For each
trajectory, reference peaks define cycle boundaries at their
midpoints. Within each cycle, both true and fitted feedback are divided by the
same maximum of the two curves. This retains their relative pulse heights and
timing while removing the large change in absolute amplitude from the initial
transient to later cycles. It is a comparison of pulse shape, not absolute
feedback magnitude. The fitted terms often underestimate feedback pulse peaks.
This comparison includes errors in the identified $z$ rollout as well as errors
in the fitted feedback function.

S6 uses the saved held-out Goodwin trajectories to evaluate each formulation's
equation at the **true** parameters, without fitting new coefficients. The strong
mismatch is the numerical derivative minus the true vector field evaluated on
the matching raw or SG-preprocessed states. The weak mismatch is the implemented
left-hand-side integral minus the implemented right-hand-side integral of the
true vector field. Its polynomial test-function degree, window length, and
stride are taken from each production-selected weak run; the held-out states
are used only to evaluate the mismatch. The additional Strong FD curve uses
second-order finite differences of the raw observations at every noise level.
Each state mismatch is divided by the RMS of its exact clean derivative, then
combined by RMS across states and both test trajectories. The solid curves in
panel b are medians over five seeds; faint points are individual seeds. Panel a
shows the absolute $x$-equation mismatch over the first 15 time units for the
first test trajectory of the 0%-noise, seed-11 runs. Strong constraints are
pointwise, while weak constraints are plotted at window centers. In panel a,
the raw weak first center is at $t=4.8$ (97-sample window), and the SG weak
first center is at $t=3.6$ (73-sample window), with $\Delta t=0.1$. Both first
windows begin at $t=0$, so the early observations are included. Consequently, S6
describes the errors in the equations supplied to each fit; it does not rank
the fitted models. In particular, the raw weak mismatch can exceed Strong + SG
at higher noise even where Figure 1 reports a better weak rollout.
The S6 ordinate differs from Figure 1a's test rollout NRMSE: Figure 1a divides
fitted-state rollout RMSE by the standard deviation of the clean states and
averages over states. S6 evaluates the true model directly on the observations,
divides equation mismatch by clean-derivative RMS, and takes an RMS over states.
Even a perfect model has a nonzero S6 mismatch when noisy observations are
inserted into its equation, while its clean rollout can be nearly exact. The two
numerical values therefore need not agree.

## Regenerate figures

Use the project environment (Python 3.11+, tested on 3.12) from the repository root:

```bash
python -m benchmarks.plot --check
python -m benchmarks.plot
```

The first command checks the selected files, numerical inputs, and run provenance.
The second regenerates all eight figures from numerical results into
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
for S2-S6. All eight figures regenerate without the local archive. Page-to-run
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

Figure 1 also marks a separate zero-noise **Strong FD (diagnostic)**
reference with a star in all three panels. It estimates derivatives numerically
from the clean sampled states using second-order finite differences, including
second-order one-sided differences at the endpoints. It uses the same clean
train/test split, degree-2 candidate library, and STLSQ solver as the campaign.
Its STLSQ threshold and alpha are selected from a deterministic 13-by-5
log-spaced grid over their configured bounds using validation rollout NRMSE.
The selected threshold is about 0.0316 and alpha is 0.000001. This single
diagnostic fit is not part of the 75 NSGA-II production runs or the median
curves. Its test NRMSE is 0.00336, coefficient relative error is 0.00849,
and it has no false-positive coefficients. The plotting script computes the
reference from the frozen configuration and writes its selected settings,
metrics, and coefficients to `figure_1_strong_fd_reference.json` in the output
directory. The published copy is in `figures/`.

Two saved Strong + SG test rollouts failed at 20% noise: that test-error median uses
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
