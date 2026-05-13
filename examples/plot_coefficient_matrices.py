#!/usr/bin/env python3
"""
Visualize coefficient matrices for identified models.

This script creates heatmap visualizations of the coefficient matrices
for the weak formulation across different parameter combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_coefficients(results_dir, noise_level, n_samples, formulation='weak', sol_idx=0):
    """
    Load coefficient matrix for a specific case.
    
    Parameters
    ----------
    results_dir : Path
        Results directory
    noise_level : float
        Noise level
    n_samples : int
        Number of samples
    formulation : str
        'weak' or 'strong'
    sol_idx : int
        Solution index (for Pareto front)
        
    Returns
    -------
    dict
        Dictionary with coefficients and metadata
    """
    dir_name = f"noise_{noise_level:.3f}_samples_{n_samples}"
    case_dir = results_dir / dir_name
    
    coef_file = case_dir / f"model_coefficients_{formulation}_sol{sol_idx}.npy"
    
    if not coef_file.exists():
        return None
    
    coefficients = np.load(coef_file)
    
    # Load metrics if available
    metrics_file = case_dir / 'metrics.json'
    rmse = None
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            if 'formulations' in metrics_data and formulation in metrics_data['formulations']:
                rmse = metrics_data['formulations'][formulation].get('metrics', {}).get('rmse')
    
    return {
        'coefficients': coefficients,
        'noise_level': noise_level,
        'n_samples': n_samples,
        'formulation': formulation,
        'rmse': rmse,
        'sol_idx': sol_idx
    }


def plot_coefficient_heatmap(coef_data, output_dir, library_terms=None):
    """
    Plot coefficient matrix as a heatmap.
    
    Parameters
    ----------
    coef_data : dict
        Coefficient data from load_coefficients
    output_dir : Path
        Output directory
    library_terms : list, optional
        Names of library terms
    """
    if coef_data is None:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coefficients = coef_data['coefficients']
    n_terms, n_vars = coefficients.shape
    
    # Default library term names if not provided
    if library_terms is None:
        library_terms = [f'Term {i}' for i in range(n_terms)]
    
    # Variable names
    var_names = ['X', 'Y', 'Z'][:n_vars]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Create heatmap
    im = ax.imshow(coefficients, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.max(np.abs(coefficients)), vmax=np.max(np.abs(coefficients)))
    
    # Set ticks and labels
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(var_names, fontsize=12, fontweight='bold')
    ax.set_yticks(range(n_terms))
    ax.set_yticklabels(library_terms, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coefficient Value', fontsize=11, fontweight='bold')
    
    # Add text annotations for non-zero values
    for i in range(n_terms):
        for j in range(n_vars):
            val = coefficients[i, j]
            if abs(val) > 1e-10:  # Only show non-zero
                text_color = 'white' if abs(val) > 0.5 * np.max(np.abs(coefficients)) else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    # Title
    rmse_str = f", RMSE: {coef_data['rmse']:.6f}" if coef_data['rmse'] is not None else ""
    ax.set_title(f"Coefficient Matrix - {coef_data['formulation'].capitalize()} Formulation\\n" +
                 f"Noise: {coef_data['noise_level']:.3f}, Samples: {coef_data['n_samples']}{rmse_str}",
                 fontsize=13, fontweight='bold', pad=15)
    
    ax.set_xlabel('State Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Library Terms', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(n_vars) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_terms) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    filename = f"coefficients_{coef_data['formulation']}_noise_{coef_data['noise_level']:.3f}_samples_{coef_data['n_samples']}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_coefficient_comparison(cases_data, output_dir):
    """
    Create a comparison plot showing coefficient matrices for multiple cases.
    
    Parameters
    ----------
    cases_data : list
        List of coefficient data dictionaries
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_cases = len(cases_data)
    if n_cases == 0:
        return
    
    # Get dimensions
    n_terms, n_vars = cases_data[0]['coefficients'].shape
    var_names = ['X', 'Y', 'Z'][:n_vars]
    
    # Create subplots
    n_cols = min(3, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 8*n_rows))
    if n_cases == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Find global min/max for consistent color scale
    all_coeffs = np.concatenate([cd['coefficients'].flatten() for cd in cases_data])
    vmax = np.max(np.abs(all_coeffs))
    
    for idx, coef_data in enumerate(cases_data):
        ax = axes[idx]
        coefficients = coef_data['coefficients']
        
        # Create heatmap
        im = ax.imshow(coefficients, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Set ticks
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(var_names, fontsize=10, fontweight='bold')
        ax.set_yticks(range(n_terms))
        ax.set_yticklabels([f'T{i}' for i in range(n_terms)], fontsize=7)
        
        # Title
        rmse_str = f"\\nRMSE: {coef_data['rmse']:.6f}" if coef_data['rmse'] is not None else ""
        ax.set_title(f"Noise: {coef_data['noise_level']:.3f}, Samples: {coef_data['n_samples']}{rmse_str}",
                    fontsize=11, fontweight='bold')
        
        # Add grid
        ax.set_xticks(np.arange(n_vars) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_terms) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Hide extra subplots
    for idx in range(n_cases, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    fig.colorbar(im, ax=axes, label='Coefficient Value', fraction=0.02, pad=0.04)
    
    plt.suptitle('Coefficient Matrix Comparison - Weak Formulation', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'coefficients_comparison_weak.png', dpi=300, bbox_inches='tight')
    print(f"Saved: coefficients_comparison_weak.png")
    plt.close()


def plot_sparsity_pattern(cases_data, output_dir):
    """
    Plot sparsity patterns (binary: zero vs non-zero).
    
    Parameters
    ----------
    cases_data : list
        List of coefficient data dictionaries
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_cases = len(cases_data)
    if n_cases == 0:
        return
    
    # Get dimensions
    n_terms, n_vars = cases_data[0]['coefficients'].shape
    var_names = ['X', 'Y', 'Z'][:n_vars]
    
    # Create subplots
    n_cols = min(3, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 7*n_rows))
    if n_cases == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, coef_data in enumerate(cases_data):
        ax = axes[idx]
        coefficients = coef_data['coefficients']
        
        # Binary sparsity pattern
        sparsity = (np.abs(coefficients) > 1e-10).astype(int)
        
        # Create heatmap
        im = ax.imshow(sparsity, aspect='auto', cmap='Greys', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(var_names, fontsize=10, fontweight='bold')
        ax.set_yticks(range(n_terms))
        ax.set_yticklabels([f'T{i}' for i in range(n_terms)], fontsize=7)
        
        # Count non-zero
        n_nonzero = np.sum(sparsity)
        n_total = sparsity.size
        
        # Title
        ax.set_title(f"Noise: {coef_data['noise_level']:.3f}, Samples: {coef_data['n_samples']}\\n" +
                    f"Non-zero: {n_nonzero}/{n_total} ({100*n_nonzero/n_total:.1f}%)",
                    fontsize=11, fontweight='bold')
        
        # Add grid
        ax.set_xticks(np.arange(n_vars) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_terms) - 0.5, minor=True)
        ax.grid(which='minor', color='red', linestyle='-', linewidth=1)
    
    # Hide extra subplots
    for idx in range(n_cases, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Sparsity Patterns - Weak Formulation', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'sparsity_patterns_weak.png', dpi=300, bbox_inches='tight')
    print(f"Saved: sparsity_patterns_weak.png")
    plt.close()


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize coefficient matrices')
    parser.add_argument('--results-dir', type=str, default='./results_cluster',
                        help='Directory containing results')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("COEFFICIENT MATRIX VISUALIZATION")
    print("="*70)
    print()
    
    # Define cases to visualize
    cases = [
        (0.000, 2500, 'Low noise, low samples'),
        (0.000, 5000, 'Low noise, medium samples'),
        (0.000, 10000, 'Low noise, high samples'),
        (0.050, 5000, 'Low noise, medium samples'),
        (0.150, 5000, 'Medium noise, medium samples'),
        (0.250, 5000, 'High noise, medium samples'),
    ]
    
    print(f"Loading coefficients for {len(cases)} cases...")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load all coefficient data
    cases_data = []
    for noise, samples, description in cases:
        print(f"Loading: {description} (noise={noise:.3f}, samples={samples})")
        coef_data = load_coefficients(results_dir, noise, samples, formulation='weak', sol_idx=0)
        
        if coef_data is not None:
            cases_data.append(coef_data)
            
            # Create individual heatmap
            plot_coefficient_heatmap(coef_data, output_dir)
        else:
            print(f"  Warning: No coefficient data found")
    
    print()
    print("-"*70)
    print("Creating comparison plots...")
    print("-"*70)
    
    # Create comparison plots
    if cases_data:
        plot_coefficient_comparison(cases_data, output_dir)
        plot_sparsity_pattern(cases_data, output_dir)
    
    print()
    print("="*70)
    print("Coefficient visualization complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
