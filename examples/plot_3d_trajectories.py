"""
3D Trajectory Visualization for Weak Formulation Results

This script creates 3D trajectory plots for the weak formulation results
from the parametric study, comparing predicted vs true trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_trajectory(trajectory_file: Path) -> np.ndarray:
    """
    Load trajectory data from .npy file.
    
    Args:
        trajectory_file: Path to the trajectory .npy file
        
    Returns:
        Trajectory array
    """
    return np.load(trajectory_file)


def plot_3d_trajectory_single(noise_level: float, n_samples: int, 
                              results_dir: Path, output_dir: Path) -> None:
    """
    Plot 3D trajectory for a single parameter combination.
    
    Args:
        noise_level: Noise level for this run
        n_samples: Number of samples for this run
        results_dir: Path to results_cluster directory
        output_dir: Directory to save the figure
    """
    # Construct directory name
    dir_name = f"noise_{noise_level:.3f}_samples_{n_samples}"
    param_dir = results_dir / dir_name
    
    # Load weak formulation trajectory
    traj_file = param_dir / 'trajectory_weak.npy'
    if not traj_file.exists():
        print(f"Warning: {traj_file} not found, skipping...")
        return
    
    trajectory = load_trajectory(traj_file)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            linewidth=1.5, alpha=0.8, color='steelblue')
    
    # Mark start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
              color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
              color='red', s=100, marker='s', label='End', zorder=5)
    
    # Labels and title
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(f'3D Trajectory - Weak Formulation\nNoise: {noise_level:.3f}, Samples: {n_samples}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'trajectory_3d_noise_{noise_level:.3f}_samples_{n_samples}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_3d_trajectory_comparison(results_dir: Path, output_dir: Path) -> None:
    """
    Create a comparison plot showing multiple trajectories for different noise levels.
    
    Args:
        results_dir: Path to results_cluster directory
        output_dir: Directory to save the figure
    """
    # Select a fixed sample size and vary noise levels
    n_samples = 5000
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, noise_level in enumerate(noise_levels, 1):
        dir_name = f"noise_{noise_level:.3f}_samples_{n_samples}"
        param_dir = results_dir / dir_name
        
        traj_file = param_dir / 'trajectory_weak.npy'
        if not traj_file.exists():
            print(f"Warning: {traj_file} not found, skipping...")
            continue
        
        trajectory = load_trajectory(traj_file)
        
        # Create subplot
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                linewidth=1.2, alpha=0.8, color='steelblue')
        
        # Mark start point
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color='green', s=80, marker='o', zorder=5)
        
        # Labels and title
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'Noise: {noise_level:.3f}', fontsize=11, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'3D Trajectories - Weak Formulation (Samples: {n_samples})', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    filename = f'trajectory_3d_comparison_samples_{n_samples}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_3d_trajectory_sample_comparison(results_dir: Path, output_dir: Path) -> None:
    """
    Create a comparison plot showing trajectories for different sample sizes at fixed noise.
    
    Args:
        results_dir: Path to results_cluster directory
        output_dir: Directory to save the figure
    """
    # Select a fixed noise level and vary sample sizes
    noise_level = 0.0
    sample_sizes = [2500, 3000, 3500, 4000, 4500, 5000]
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, n_samples in enumerate(sample_sizes, 1):
        dir_name = f"noise_{noise_level:.3f}_samples_{n_samples}"
        param_dir = results_dir / dir_name
        
        traj_file = param_dir / 'trajectory_weak.npy'
        if not traj_file.exists():
            print(f"Warning: {traj_file} not found, skipping...")
            continue
        
        trajectory = load_trajectory(traj_file)
        
        # Create subplot
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                linewidth=1.2, alpha=0.8, color='steelblue')
        
        # Mark start point
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color='green', s=80, marker='o', zorder=5)
        
        # Labels and title
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'Samples: {n_samples}', fontsize=11, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'3D Trajectories - Weak Formulation (Noise: {noise_level:.3f})', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    filename = f'trajectory_3d_comparison_noise_{noise_level:.3f}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    """Main visualization workflow."""
    # Set up paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results_cluster'
    output_dir = results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading trajectories from: {results_dir}")
    print(f"Saving visualizations to: {output_dir}\n")
    
    print("Generating 3D trajectory visualizations...")
    print("-" * 80)
    
    # Create comparison plots
    print("\nCreating noise level comparison (fixed samples=5000)...")
    plot_3d_trajectory_comparison(results_dir, output_dir)
    
    print("\nCreating sample size comparison (fixed noise=0.0)...")
    plot_3d_trajectory_sample_comparison(results_dir, output_dir)
    
    # Create individual plots for best and worst cases
    print("\nCreating individual trajectory plots...")
    
    # Best case: noise=0.0, samples=5000
    plot_3d_trajectory_single(0.0, 5000, results_dir, output_dir)
    
    # High noise case: noise=0.25, samples=5000
    plot_3d_trajectory_single(0.25, 5000, results_dir, output_dir)
    
    # Low samples case: noise=0.0, samples=2500
    plot_3d_trajectory_single(0.0, 2500, results_dir, output_dir)
    
    print("-" * 80)
    print(f"\nVisualization complete! All figures saved to: {output_dir}")
    
    # Count total PNG files
    total_pngs = len(list(output_dir.glob('trajectory_3d*.png')))
    print(f"Total 3D trajectory figures generated: {total_pngs}")


if __name__ == '__main__':
    main()
