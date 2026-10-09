# Paper benchmarks

The selected Figure 1 and Figure 2 results are frozen in
[`data/current/`](data/current/) with their run JSON, saved trajectories,
configurations, provenance, and SHA-256 manifest. The final main and A4
supplementary figures are in [`figures/`](figures/).

## Verify the frozen results

From the repository root, in the project environment:

```bash
python -m benchmarks.freeze
```

This checks every tracked campaign file against the manifest, the complete
75-run Figure 1 and 10-run Figure 2 job matrices, evaluation budgets, saved
trajectory scores, main figure hashes, and the A4 supplementary page lists.
The Figure 1 PDF has one page per noise/seed pair, with all three selected
models against the same noisy test data. The Figure 2 PDF has one page per
selected joint-library run. To render the A4 pages from saved data:

```bash
python -m benchmarks.si_paper
```

Rendered copies go to `.local/si_paper_render/`. To replace a frozen figure,
copy the reviewed output to `figures/` and regenerate the checksum manifest
with `python -m benchmarks.freeze --write-manifest`.

The 25 strong-form Figure 1 baselines were reused from the earlier campaign;
the matching trajectories and their origin are recorded in
[`data/current/noise/strong_trajectory_provenance.json`](data/current/noise/strong_trajectory_provenance.json).
The 50 corrected weak-form and 10 joint-library run JSON records have
`git_dirty=true` in their original provenance. Their saved weak-form source
hash matches this repository, and the stored configuration and artifacts are
checked by the freeze command. A separate clean Figure 2 block 4 run is in
[`data/current/verification/`](data/current/verification/); it reproduced the
selected model, coefficients, settings, and test trajectory exactly. This
does not claim an independent rerun of all 60 corrected runs.

## Historical audit data

[`data/historical/`](data/historical/) contains the earlier production JSON
results and saved trajectories, generated before the corrected weak-form
weight calculation. Its manifest records checksums and source commits. Check
the 85 archived records with:

```bash
python -m benchmarks.plot --check
```

To render historical figures for comparison, run
`python -m benchmarks.plot --historical`. Output goes to the ignored
`.local/benchmark_plots/historical/` directory.

## Run new campaigns

From a clean, committed source tree:

```bash
python -m benchmarks.run noise --workers 5
python -m benchmarks.run joint --workers 5
```

Each command runs or resumes a campaign under the ignored
`.local/benchmark_runs/<experiment>/` directory. A cached job is reused only
when its commit, config hash, artifact, and evaluation budget match.

| Experiment | Design | Budget |
|---|---|---|
| Noise | Three methods, five noise levels, five seeds; 3/1/2 train/validation/test split | 75 × 2,400 evaluations |
| Joint | Ten repetitions at 10% noise; 2/1/2 split; Latin-hypercube initial conditions | 10 × 2,400 evaluations |

Both campaigns minimize trajectory RMSE, Durbin–Watson deviation from two,
and active coefficient count. Joint optimization searches Hill and exponential
library parameters, weak-window settings, and STLSQ settings. The settings
are in [`noise/config.toml`](noise/config.toml) and
[`joint/config.toml`](joint/config.toml).

The [Figure 2 reproduction tutorial](../docs/tutorials/goodwin_joint.md)
walks through one joint run.
