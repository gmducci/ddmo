# Figure 3 — Joint parametric-library Goodwin optimization

This campaign demonstrates SG-preprocessed optimized weak identification at
10% noise. The candidate library contains all polynomials through degree two,
pairwise interactions, a Hill family parameterized by `gamma_1`, and an
exponential family parameterized by `gamma_2`.

The seven outer variables are `gamma_1`, `gamma_2`, weak-test-function degree,
window length, stride, STLSQ threshold, and STLSQ alpha. STLSQ fits and
sparsifies the 45 equation coefficients for every outer candidate.

The Goodwin ground truth contains the Hill term with `gamma_1 = 9.5` but no
exponential term. Consequently `gamma_2` is an optimized nuisance parameter,
not an identifiable recovery target when exponential coefficients are zero.

## Current status

- Experiment configuration: ready.
- Job planning: ready through `benchmarks.common.plan_runs`.
- Scientific runner: complete for five production seeds.
- Trajectory artifact writer: ready and verified.
- Postprocessor: ready; generates the held-out rollout and parameter panel.

The runner saves selected train, validation, and test trajectories alongside
scalar metrics so the final figure can show observed, reference, and model
rollouts.

Inspect the planned five-job matrix with:

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_3/config.toml
```

Run its smoke job with:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_3.run \
  --job-id joint_goodwin_poly2_hill_exp_sg-weak_joint_sg-noise010-seed011 \
  --smoke
```

Generate the current figure preview with:

```bash
MPLBACKEND=Agg python -m benchmarks.figures.figure_3.postprocess
```

After committing all benchmark code, run or resume the strict publication
campaign with:

```bash
python -m benchmarks.figures.figure_3.run_campaign --workers 5
```
