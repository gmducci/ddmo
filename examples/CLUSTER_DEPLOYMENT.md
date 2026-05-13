# Cluster Deployment Guide

## Summary

Your parametric study script now supports **distributing grid points across fewer jobs**!

- **Grid**: 6 noise levels × 6 sample counts = **36 grid points**
- **Jobs**: Can run with any number of jobs (e.g., 10 jobs for 36 points)
- **Distribution**: Automatic and balanced

## How It Works

### Example: 36 Grid Points with 10 Jobs

```
Job 0: processes grid points 0-3   (4 points)
Job 1: processes grid points 4-7   (4 points)
Job 2: processes grid points 8-11  (4 points)
...
Job 6: processes grid points 24-26 (3 points)
Job 7: processes grid points 27-29 (3 points)
Job 8: processes grid points 30-32 (3 points)
Job 9: processes grid points 33-35 (3 points)
```

Each job processes 3-4 grid points automatically!

## Usage

### 1. Visualize the Distribution

```bash
python show_job_distribution.py 10
```

This shows exactly which grid points each job will process.

### 2. Submit to Cluster

```bash
sbatch submit_parametric_study.sh
```

The script is already configured for:
- 10 jobs (`--array=0-9`)
- 36 grid points distributed across them
- Your cluster environment (waterboa module, etc.)

### 3. Adjust for Different Grid Sizes

If you change the grid in `define_parameter_grid()`:

```python
noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25]  # 6 values
n_samples = [2500, 3000, 3500, 4000, 4500, 5000]  # 6 values
# Total: 6 × 6 = 36 grid points
```

Then update the SLURM script:
- For 100 grid points with 10 jobs: `#SBATCH --array=0-9`
- For 100 grid points with 20 jobs: `#SBATCH --array=0-19`
- For 100 grid points with 100 jobs: `#SBATCH --array=0-99`

## Key Parameters

### In Python Script
- `--job-id`: SLURM array task ID (automatically set)
- `--total-jobs`: Total number of jobs (set to match array size)

### In SLURM Script
- `#SBATCH --array=0-9`: Creates 10 jobs
- `TOTAL_JOBS=10`: Must match array size

## Results

Each grid point saves results to:
```
results_cluster/
  noise_0.000_samples_2500/
  noise_0.000_samples_3000/
  ...
  noise_0.250_samples_5000/
```

All results are saved with proper grid indices and seeds for reproducibility!

## Testing Locally

```bash
# Test job 0 (processes 4 grid points)
python ex_goodwin_parametric_study.py --job-id 0 --total-jobs 10

# Test job 5 (processes 4 grid points)
python ex_goodwin_parametric_study.py --job-id 5 --total-jobs 10
```
