# Goodwin Model Parametric Study

This document explains how to use `ex_goodwin_parametric_study.py` for systematic comparison of strong vs weak formulations.

## Overview

The script performs a 2D parametric study varying:
- **Noise levels**: Amount of measurement noise added to the data
- **Number of samples**: Length of the time series

For each grid point, both **strong** and **weak** formulations are evaluated on **identical data** (same random seed) to ensure fair comparison.

## Parameter Grid Definition

The parameter grid is **hardcoded in the script** in the `define_parameter_grid()` function. To modify the parameter space, edit this function:

```python
def define_parameter_grid():
    # Define your parameter grid here
    noise_levels = [0.05, 0.10, 0.15, 0.20]
    n_samples = [1000, 2000, 3000, 5000]
    
    return {
        'noise_levels': noise_levels,
        'n_samples': n_samples
    }
```

This creates a 4×4 = **16 grid points** (cartesian product).

## Visualizing the Grid

Use the helper script to see the job ID mapping:

```bash
python show_grid_mapping.py
```

This shows which job ID corresponds to which (noise, samples) pair.

## What Gets Optimized

### Strong Formulation
- Hill coefficient (`n_hill`)
- Exponential parameter (`gamma_2`)
- STLSQ threshold
- STLSQ alpha (regularization)

### Weak Formulation
- Hill coefficient (`n_hill`)
- Exponential parameter (`gamma_2`)
- STLSQ threshold
- STLSQ alpha (regularization)
- Window length
- Stride

## Basic Usage

### Local Testing

```bash
# Run all grid points sequentially (for testing)
python ex_goodwin_parametric_study.py

# Run a specific grid point (e.g., job ID 5)
python ex_goodwin_parametric_study.py --job-id 5

# Run with custom optimization settings
python ex_goodwin_parametric_study.py --job-id 5 --n-gen 20 --pop-size 200

# Run only strong formulation
python ex_goodwin_parametric_study.py --job-id 5 --formulations strong
```

### Cluster Deployment (Job Arrays)

The recommended way to run on a cluster is using SLURM job arrays:

```bash
# Submit all grid points as a job array
sbatch submit_parametric_study.sh

# Or specify the array range explicitly
sbatch --array=0-15 submit_parametric_study.sh

# Run only a subset (e.g., first 4 grid points)
sbatch --array=0-3 submit_parametric_study.sh
```

The `submit_parametric_study.sh` script automatically passes `$SLURM_ARRAY_TASK_ID` as `--job-id` to the Python script.

## Command-Line Arguments

### Required for Cluster
- `--job-id`: Job array index (0-based). If not specified, runs all grid points sequentially.

### Optional Settings
- `--n-gen`: Number of generations for NSGA3 (default: `10`)
- `--pop-size`: Population size for NSGA3 (default: `100`)
- `--output-dir`: Directory for results (default: `./results_parametric`)
- `--seed-base`: Base random seed for reproducibility (default: `42`)
- `--formulations`: Which formulations to test (default: `strong weak`)
  - Options: `strong`, `weak`, or both

## Output Structure

Results are organized as follows:

```
results_parametric/
├── noise_0.050_samples_1000/
│   ├── results.pkl          # Full results (pickled)
│   └── metrics.json         # Summary metrics
├── noise_0.050_samples_2000/
│   ├── results.pkl
│   └── metrics.json
└── ...
```

### Metrics JSON Structure

```json
{
  "noise_level": 0.05,
  "n_samples": 1000,
  "formulations": {
    "strong": {
      "metrics": {
        "mse": 0.0123,
        "rmse": 0.111,
        "rmse_x": 0.05,
        "rmse_y": 0.08,
        "rmse_z": 0.12,
        "normalized_rmse": [0.02, 0.03, 0.04]
      },
      "computation_time": 245.3
    },
    "weak": {
      "metrics": { ... },
      "computation_time": 312.7
    }
  }
}
```

## Key Features

### 1. Reproducibility
- Each grid point uses a unique but deterministic random seed
- Same data is used for both formulations at each grid point
- Seed = `seed_base + grid_point_index`

### 2. Fair Comparison
- Identical noisy data for strong and weak formulations
- Same library of functions
- Same optimization algorithm (NSGA3)
- Only formulation-specific parameters differ

### 3. Cluster-Ready
- Non-interactive matplotlib backend
- No `plt.show()` calls
- Results saved to disk automatically
- Job array support for parallel execution

### 4. Comprehensive Metrics
- MSE and RMSE (overall and per-variable)
- Normalized RMSE (relative to signal range)
- Computation time
- Full result objects saved for post-processing

## Next Steps

After running the parametric study, you can:

1. **Aggregate results** across all grid points
2. **Create comparison plots** (heatmaps of RMSE vs noise/samples)
3. **Analyze Pareto fronts** for each formulation
4. **Extract identified coefficients** and compare to ground truth

Would you like me to create analysis scripts for these tasks?

## Example Analysis Script

```python
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Load all metrics
results_dir = Path('./results_parametric')
all_metrics = []

for subdir in sorted(results_dir.glob('noise_*_samples_*')):
    with open(subdir / 'metrics.json', 'r') as f:
        all_metrics.append(json.load(f))

# Extract data for heatmap
noise_levels = sorted(set(m['noise_level'] for m in all_metrics))
sample_counts = sorted(set(m['n_samples'] for m in all_metrics))

# Create RMSE comparison heatmaps
# ... (to be implemented)
```
