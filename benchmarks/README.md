# Publication benchmarks

The benchmark code is organized by the figure it produces in the SoftwareX
manuscript. Each figure directory owns its frozen configuration, scientific
runner, postprocessor, and usage notes. Shared utilities live in `common/`, and
generated artifacts remain together under `results/`.

```text
benchmarks/
├── common/
│   ├── plan_runs.py
│   └── style.py
├── figures/
│   ├── figure_1/
│   │   ├── config.toml
│   │   ├── run.py
│   │   ├── postprocess.py
│   │   └── README.md
│   ├── figure_2/
│   │   ├── config.toml
│   │   ├── run.py
│   │   └── README.md
│   └── figure_3/
│       ├── config.toml
│       ├── run.py
│       └── README.md
└── results/
    ├── raw/
    ├── aggregated/
    └── figures/
```

## Manuscript mapping

| Folder | Intended figure | Status |
|---|---|---|
| `figure_1` | Brusselator parameter recovery and predictive accuracy | Runner and postprocessor ready |
| `figure_2` | Goodwin noise and formulation robustness | Runner and postprocessor ready |
| `figure_3` | Complete joint Goodwin optimization | Runner, artifacts, and postprocessor ready |

The individual figure README is the authoritative run guide and explicitly
lists anything still missing. This prevents a configuration-only campaign from
being mistaken for an executable benchmark.

## Data split

Every experiment is designed around three training trajectories, one separate
validation trajectory for Pareto-model selection, and two untouched test
trajectories. Initial conditions must be frozen in the corresponding figure
configuration before production execution.

## Shared job planning

Inspect any job matrix without running optimization:

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_1/config.toml
```

Write a manifest when needed:

```bash
python -m benchmarks.common.plan_runs \
  benchmarks/figures/figure_2/config.toml \
  --output benchmarks/results/figure_2_jobs.csv
```

## Output contract

Each completed job must record:

- Git commit and dirty-state flag;
- package and dependency versions;
- experiment, method, noise level, and seed;
- optimizer budget and actual candidate evaluations;
- full Pareto objectives and decision variables;
- selected coefficients and symbolic equations;
- validation and test metrics;
- failed-candidate count and wall time.

Raw outputs go under `results/raw/`, aggregated tables under
`results/aggregated/`, and PNG/PDF figures under `results/figures/`. Generated
results are ignored by Git because archival campaign data should be deposited
separately from the source distribution.
