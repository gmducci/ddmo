#!/usr/bin/env python3
"""
Phase Space Comparison: Identified Models vs Ground Truth

This script creates phase space plots comparing the identified model trajectories
with the ground truth (clean) data for selected parameter combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json


def load_case_data(results_dir, noise_level, n_samples):
    """
    Load data for a specific parameter combination.
    
    Parameters
    ----------
    results_dir : Path
        Results directory
    noise_level : float
        Noise level
    n_samples : int
        Number of samples
        
    Returns
    -------
    dict
        Dictionary with clean data, noisy data, and trajectories
    """
    dir_name = f"noise_{noise_level:.3f}_samples_{n_samples}"
    case_dir = results_dir / dir_name
    
    data = {
        'noise_level': noise_level,
        'n_samples': n_samples,
        'case_dir': case_dir
    }
    
    # Load clean and noisy data
    if (case_dir / 'data_clean.npy').exists():
        data['clean'] = np.load(case_dir / 'data_clean.npy')
    
    if (case_dir / 'data_noisy.npy').exists():
        data['noisy'] = np.load(case_dir / 'data_noisy.npy')
    
    if (case_dir / 'time_array.npy').exists():
        data['time'] = np.load(case_dir / 'time_array.npy')
    
    # Load trajectories for both formulations
    for formulation in ['strong', 'weak']:
        traj_file = case_dir / f'trajectory_{formulation}.npy'
        if traj_file.exists():
            data[f'trajectory_{formulation}'] = np.load(traj_file)
    
    # Load metrics
    metrics_file = case_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)
    
    return data


def plot_2d_phase_space(data, formulation, var_pairs, output_dir):
    """
    Create 2D phase space plots for specified variable pairs.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_case_data
    formulation : str
        'strong' or 'weak'
    var_pairs : list of tuples
        List of (var1_idx, var2_idx, var1_name, var2_name) tuples
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    traj_key = f'trajectory_{formulation}'
    if traj_key not in data or 'clean' not in data:
        print(f"Warning: Missing data for {formulation} formulation")
        return
    
    trajectory = data[traj_key]
    clean = data['clean']
    noisy = data.get('noisy', None)
    
    # Match lengths
    min_len = min(len(trajectory), len(clean))
    trajectory = trajectory[:min_len]
    clean = clean[:min_len]
    if noisy is not None:
        noisy = noisy[:min_len]
    
    n_pairs = len(var_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    
    for idx, (i, j, name_i, name_j) in enumerate(var_pairs):
        ax = axes[idx]
        
        # Plot noisy data (training data) if available
        if noisy is not None:
            ax.scatter(noisy[:, i], noisy[:, j], c='gray', s=10, alpha=0.3, label='Noisy Data (Training)', zorder=1)
        
        # Plot ground truth
        ax.plot(clean[:, i], clean[:, j], 'k-', linewidth=2.5, alpha=0.8, label='Ground Truth', zorder=3)
        
        # Plot identified model
        ax.plot(trajectory[:, i], trajectory[:, j], 'r--', linewidth=2, alpha=0.8, label='Identified Model', zorder=2)
        
        # Mark start points
        ax.plot(clean[0, i], clean[0, j], 'go', markersize=10, label='Start', zorder=5)
        
        ax.set_xlabel(name_i, fontsize=12, fontweight='bold')
        ax.set_ylabel(name_j, fontsize=12, fontweight='bold')
        ax.set_title(f'{name_i} vs {name_j}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Get RMSE for title
    rmse = 'N/A'
    if 'metrics' in data and 'formulations' in data['metrics']:
        if formulation in data['metrics']['formulations']:
            metrics = data['metrics']['formulations'][formulation].get('metrics', {})
            if 'rmse' in metrics:
                rmse = f"{metrics['rmse']:.6f}"
    
    fig.suptitle(f'Phase Space - {formulation.capitalize()} Formulation\\n' +
                 f'Noise: {data["noise_level"]:.3f}, Samples: {data["n_samples"]}, RMSE: {rmse}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    filename = f'phase_space_2d_{formulation}_noise_{data["noise_level"]:.3f}_samples_{data["n_samples"]}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_3d_phase_space(data, formulation, output_dir):
    """
    Create 3D phase space plot.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_case_data
    formulation : str
        'strong' or 'weak'
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    traj_key = f'trajectory_{formulation}'
    if traj_key not in data or 'clean' not in data:
        print(f"Warning: Missing data for {formulation} formulation")
        return
    
    trajectory = data[traj_key]
    clean = data['clean']
    
    # Match lengths
    min_len = min(len(trajectory), len(clean))
    trajectory = trajectory[:min_len]
    clean = clean[:min_len]
    
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(clean[:, 0], clean[:, 1], clean[:, 2], 'k-', linewidth=2, alpha=0.7, label='Ground Truth')
    ax1.scatter(clean[0, 0], clean[0, 1], clean[0, 2], color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.set_xlabel('X', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Identified model
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r--', linewidth=2, alpha=0.7, label='Identified Model')
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, marker='o', label='Start', zorder=5)
    ax2.set_xlabel('X', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax2.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax2.set_title('Identified Model', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.3)
    
    # Get RMSE for title
    rmse = 'N/A'
    if 'metrics' in data and 'formulations' in data['metrics']:
        if formulation in data['metrics']['formulations']:
            metrics = data['metrics']['formulations'][formulation].get('metrics', {})
            if 'rmse' in metrics:
                rmse = f"{metrics['rmse']:.6f}"
    
    fig.suptitle(f'3D Phase Space - {formulation.capitalize()} Formulation\\n' +
                 f'Noise: {data["noise_level"]:.3f}, Samples: {data["n_samples"]}, RMSE: {rmse}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    filename = f'phase_space_3d_{formulation}_noise_{data["noise_level"]:.3f}_samples_{data["n_samples"]}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_3d_overlay(data, formulation, output_dir):
    """
    Create 3D phase space plot with ground truth, noisy data, and identified model overlaid.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_case_data
    formulation : str
        'strong' or 'weak'
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    traj_key = f'trajectory_{formulation}'
    if traj_key not in data or 'clean' not in data:
        print(f"Warning: Missing data for {formulation} formulation")
        return
    
    trajectory = data[traj_key]
    clean = data['clean']
    noisy = data.get('noisy', None)
    
    # Match lengths
    min_len = min(len(trajectory), len(clean))
    trajectory = trajectory[:min_len]
    clean = clean[:min_len]
    if noisy is not None:
        noisy = noisy[:min_len]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot noisy data (training data) if available
    if noisy is not None:
        ax.scatter(noisy[:, 0], noisy[:, 1], noisy[:, 2], c='gray', s=5, alpha=0.2, label='Noisy Data (Training)', zorder=1)
    
    # Plot both trajectories
    ax.plot(clean[:, 0], clean[:, 1], clean[:, 2], 'k-', linewidth=2.5, alpha=0.8, label='Ground Truth', zorder=3)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r--', linewidth=2, alpha=0.8, label='Identified Model', zorder=2)
    
    # Mark start point
    ax.scatter(clean[0, 0], clean[0, 1], clean[0, 2], color='green', s=150, marker='o', label='Start', zorder=5)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    
    # Get RMSE for title
    rmse = 'N/A'
    if 'metrics' in data and 'formulations' in data['metrics']:
        if formulation in data['metrics']['formulations']:
            metrics = data['metrics']['formulations'][formulation].get('metrics', {})
            if 'rmse' in metrics:
                rmse = f"{metrics['rmse']:.6f}"
    
    plt.title(f'3D Phase Space Overlay - {formulation.capitalize()} Formulation\\n' +
              f'Noise: {data["noise_level"]:.3f}, Samples: {data["n_samples"]}, RMSE: {rmse}',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    filename = f'phase_space_3d_overlay_{formulation}_noise_{data["noise_level"]:.3f}_samples_{data["n_samples"]}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create phase space comparison plots')
    parser.add_argument('--results-dir', type=str, default='./results_cluster',
                        help='Directory containing results')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("PHASE SPACE COMPARISON ANALYSIS")
    print("="*70)
    print()
    
    # Define cases to analyze
    cases = [
        (0.000, 2500, 'Low noise, low samples'),
        (0.000, 5000, 'Low noise, high samples'),
        (0.150, 5000, 'Medium noise, high samples'),
        (0.250, 5000, 'High noise, high samples'),
    ]
    
    # Define variable pairs for 2D plots
    var_pairs = [
        (0, 1, 'X', 'Y'),
        (1, 2, 'Y', 'Z'),
        (0, 2, 'X', 'Z'),
    ]
    
    print(f"Analyzing {len(cases)} parameter combinations...")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    for noise, samples, description in cases:
        print(f"\\nProcessing: {description} (noise={noise:.3f}, samples={samples})")
        print("-"*70)
        
        # Load data
        data = load_case_data(results_dir, noise, samples)
        
        if 'clean' not in data:
            print(f"  Warning: No clean data found, skipping...")
            continue
        
        # Process both formulations
        for formulation in ['weak', 'strong']:
            if f'trajectory_{formulation}' not in data:
                print(f"  Warning: No {formulation} trajectory found, skipping...")
                continue
            
            print(f"\\n  {formulation.capitalize()} formulation:")
            
            # Create 2D phase space plots
            plot_2d_phase_space(data, formulation, var_pairs, output_dir)
            
            # Create 3D phase space plots
            plot_3d_phase_space(data, formulation, output_dir)
            
            # Create 3D overlay plot
            plot_3d_overlay(data, formulation, output_dir)
    
    print("\\n" + "="*70)
    print("Phase space analysis complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
