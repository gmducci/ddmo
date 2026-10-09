# Paper benchmarks

The previous Figure 1, Figure 2, and supplementary artwork has been removed
from the tracked repository. It was generated before the corrected weak-form
weight calculation. New paper artwork will be frozen together after the
supplementary-figure review.

## Directory layout

```text
benchmarks/
  common/       Goodwin data generation, execution, and result validation
  noise/        Figure 1 configuration, runner, and plotting code
  joint/        Figure 2 configuration, initial conditions, runner, and plotting code
  data/         Corrected tutorial reference and historical/ campaign bundles
  diagnostics.py
  plot.py       Historical data checks and local rendering
  run.py        Production campaign entry point
```

`data/historical/noise.zip` and `data/historical/joint.zip` contain the earlier production JSON
results. Their matching `*_trajectories.npz` files hold the saved trajectories,
and `data/historical/manifest.json` records checksums and original source commits. They
are retained for audit, not presented as current corrected-weight paper results.
The [Figure 2 block 4 tutorial reference](data/figure2_tutorial_block04.json)
is a compact corrected-weight result. A clean run from commit `ab7da8a3`
reproduced its selected model, coefficients, settings, and test trajectory
exactly.

The complete corrected campaigns are currently in the ignored local folders
`.local/figure1_weak_fix/` and `.local/figure2_weak_fix/`. They contain figure
files, run JSON, trajectory NPZ, configuration, and provenance metadata. The
older source and artwork remain available in Git history. Neither local folder
is required to import or test `ddmo`.

## Check archived data

From the repository root, in the project environment:

```bash
python -m benchmarks.plot --check
```

This validates all 85 archived production records and their trajectory bundles,
then checks that the corrected tutorial reference names the current weak-form
source hash. It does not certify artwork.

To render the historical figures locally for comparison:

```bash
python -m benchmarks.plot --historical
```

Outputs go to `.local/benchmark_plots/historical/`. They are excluded from Git.
Use `noise` or `joint` to render one experiment, and `--diagnostics` or
`--library-overview` for the historical supplementary views.

## Run new campaigns

From a clean, committed source tree:

```bash
python -m benchmarks.run noise --workers 5
python -m benchmarks.run joint --workers 5
```

Each command runs or resumes its full campaign and writes new raw results,
logs, and plots under `.local/benchmark_runs/<experiment>/`. A cached job is
reused only when its commit, config hash, artifact, and evaluation budget match.

| Experiment | Design | Budget |
|---|---|---|
| Noise | Three methods, five noise levels, five seeds; 3/1/2 train/validation/test split | 75 × 2,400 evaluations |
| Joint | Ten repetitions at 10% noise; 2/1/2 split; Latin-hypercube initial conditions | 10 × 2,400 evaluations |

Both campaigns minimize trajectory RMSE, Durbin–Watson deviation from two,
and active coefficient count. Joint optimization searches Hill and exponential
library parameters, weak-window settings, and STLSQ settings. The exact
settings are in `noise/config.toml` and `joint/config.toml`.

The corrected-weight tutorial for one joint run is in
[`docs/tutorials/goodwin_joint.md`](../docs/tutorials/goodwin_joint.md).
