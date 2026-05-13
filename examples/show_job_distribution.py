#!/usr/bin/env python3
"""
Helper script to visualize how grid points are distributed across jobs
when using the --total-jobs parameter.
"""

import sys

def define_parameter_grid():
    noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
    n_samples = [2500, 3000, 3500, 4000, 4500, 5000]
    return {
        'noise_levels': noise_levels,
        'n_samples': n_samples
    }

def calculate_job_distribution(total_grid_points, total_jobs):
    """Calculate which grid points each job processes."""
    points_per_job = total_grid_points // total_jobs
    remainder = total_grid_points % total_jobs
    
    job_assignments = []
    for job_id in range(total_jobs):
        if job_id < remainder:
            start_idx = job_id * (points_per_job + 1)
            end_idx = start_idx + points_per_job + 1
        else:
            start_idx = remainder * (points_per_job + 1) + (job_id - remainder) * points_per_job
            end_idx = start_idx + points_per_job
        
        job_assignments.append((job_id, start_idx, end_idx))
    
    return job_assignments

def main():
    if len(sys.argv) > 1:
        try:
            total_jobs = int(sys.argv[1])
        except ValueError:
            print("Usage: python show_job_distribution.py [total_jobs]")
            print("Example: python show_job_distribution.py 10")
            sys.exit(1)
    else:
        total_jobs = None
    
    param_grid = define_parameter_grid()
    noise_levels = param_grid['noise_levels']
    n_samples_list = param_grid['n_samples']
    
    # Generate grid points
    grid_points = [
        (noise, n_samp) 
        for noise in noise_levels 
        for n_samp in n_samples_list
    ]
    
    total_grid_points = len(grid_points)
    
    print("="*80)
    print("GRID POINT DISTRIBUTION ACROSS JOBS")
    print("="*80)
    print(f"\nTotal grid points: {total_grid_points}")
    print(f"Noise levels: {noise_levels}")
    print(f"Sample counts: {n_samples_list}")
    print()
    
    if total_jobs is None:
        print("Usage modes:")
        print()
        print("1. One job per grid point (default):")
        print(f"   sbatch --array=0-{total_grid_points-1} submit_parametric_study.sh")
        print(f"   python ex_goodwin_parametric_study.py --job-id <0-{total_grid_points-1}>")
        print()
        print("2. Distribute across fewer jobs:")
        print(f"   python {sys.argv[0]} <total_jobs>")
        print(f"   Example: python {sys.argv[0]} 10")
        print()
        return
    
    if total_jobs > total_grid_points:
        print(f"WARNING: total_jobs ({total_jobs}) > grid points ({total_grid_points})")
        print(f"Some jobs will have nothing to process!")
        print()
    
    job_assignments = calculate_job_distribution(total_grid_points, total_jobs)
    
    print(f"Distribution: {total_grid_points} grid points across {total_jobs} jobs")
    print()
    print("-"*80)
    print(f"{'Job ID':<10} {'Grid Points':<20} {'Count':<10} {'Parameters'}")
    print("-"*80)
    
    for job_id, start_idx, end_idx in job_assignments:
        count = end_idx - start_idx
        if count > 0:
            # Show first and last point for this job
            first_point = grid_points[start_idx]
            last_point = grid_points[end_idx-1]
            
            if count == 1:
                params_str = f"noise={first_point[0]:.3f}, samples={first_point[1]}"
            else:
                params_str = f"noise={first_point[0]:.3f}...{last_point[0]:.3f}, samples={first_point[1]}...{last_point[1]}"
            
            print(f"{job_id:<10} {start_idx}-{end_idx-1:<17} {count:<10} {params_str}")
        else:
            print(f"{job_id:<10} {'(empty)':<20} {0:<10} -")
    
    print("-"*80)
    print()
    print("To run on cluster:")
    print(f"  sbatch --array=0-{total_jobs-1} submit_parametric_study.sh")
    print()
    print("Make sure to:")
    print(f"  1. Set TOTAL_JOBS={total_jobs} in submit_parametric_study.sh")
    print(f"  2. Uncomment Option 2 in the script")
    print(f"  3. Update #SBATCH --array=0-{total_jobs-1}")
    print("="*80)

if __name__ == "__main__":
    main()
