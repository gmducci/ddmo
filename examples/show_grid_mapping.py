#!/usr/bin/env python3
"""
Helper script to visualize the parameter grid mapping.
Shows which job ID corresponds to which (noise, samples) pair.
"""

# Import the grid definition from the main script
import sys
sys.path.insert(0, '.')

# Define the same grid as in the main script
def define_parameter_grid():
    noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
    n_samples = [2500, 3000, 3500, 4000, 4500, 5000]
    return {
        'noise_levels': noise_levels,
        'n_samples': n_samples
    }

def main():
    param_grid = define_parameter_grid()
    noise_levels = param_grid['noise_levels']
    n_samples_list = param_grid['n_samples']
    
    # Generate grid points (same logic as main script)
    grid_points = [
        (noise, n_samp) 
        for noise in noise_levels 
        for n_samp in n_samples_list
    ]
    
    print("="*70)
    print("PARAMETER GRID MAPPING")
    print("="*70)
    print(f"\nNoise levels: {noise_levels}")
    print(f"Sample counts: {n_samples_list}")
    print(f"\nTotal grid points: {len(grid_points)}")
    print("\n" + "-"*70)
    print(f"{'Job ID':<10} {'Noise Level':<15} {'Samples':<15} {'Seed':<10}")
    print("-"*70)
    
    seed_base = 42
    for idx, (noise, samples) in enumerate(grid_points):
        seed = seed_base + idx
        print(f"{idx:<10} {noise:<15.3f} {samples:<15} {seed:<10}")
    
    print("-"*70)
    print(f"\nTo run on cluster:")
    print(f"  sbatch --array=0-{len(grid_points)-1} submit_parametric_study.sh")
    print(f"\nTo run specific grid point locally:")
    print(f"  python ex_goodwin_parametric_study.py --job-id <ID>")
    print(f"\nTo run all grid points locally:")
    print(f"  python ex_goodwin_parametric_study.py")
    print("="*70)

if __name__ == "__main__":
    main()
