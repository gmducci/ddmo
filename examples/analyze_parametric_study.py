"""
Parametric Study Results Analysis

This script analyzes the results from a parametric study of the Goodwin model,
comparing strong and weak formulations across different noise levels and sample sizes.

The results are stored in subdirectories under 'results_cluster/', with each directory
containing metrics and trajectory data for one parameter combination.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all metrics from the parametric study results.
    
    Args:
        results_dir: Path to the results_cluster directory
        
    Returns:
        DataFrame containing all metrics from all parameter combinations
    """
    all_data = []
    
    # Iterate through all subdirectories
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Skip non-result directories (e.g., 'analysis')
        if not subdir.name.startswith('noise_'):
            continue
            
        # Parse directory name to extract parameters
        parts = subdir.name.split('_')
        noise_level = float(parts[1])
        n_samples = int(parts[3])
        
        # Load metrics from JSON file
        metrics_file = subdir / 'metrics.json'
        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found, skipping...")
            continue
            
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # Extract metrics for both formulations
        for formulation in ['strong', 'weak']:
            if formulation not in data['formulations']:
                continue
                
            form_data = data['formulations'][formulation]
            metrics = form_data['metrics']
            
            # Skip if this formulation failed (has error instead of metrics)
            if 'error' in metrics or 'mse' not in metrics:
                print(f"  Skipping {formulation} (failed): noise={noise_level}, samples={n_samples}")
                continue
            
            record = {
                'noise_level': noise_level,
                'n_samples': n_samples,
                'formulation': formulation,
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'rmse_x': metrics['rmse_x'],
                'rmse_y': metrics['rmse_y'],
                'rmse_z': metrics['rmse_z'],
                'normalized_rmse_x': metrics['normalized_rmse'][0],
                'normalized_rmse_y': metrics['normalized_rmse'][1],
                'normalized_rmse_z': metrics['normalized_rmse'][2],
                'computation_time': form_data['computation_time'],
                'n_gen': form_data['n_gen'],
                'pop_size': form_data['pop_size']
            }
            all_data.append(record)
    
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} records from {len(df) // 2} parameter combinations")
    return df


def create_heatmap(df: pd.DataFrame, metric: str, formulation: str, 
                   output_dir: Path, title_suffix: str = "") -> None:
    """
    Create a 2D heatmap showing a metric vs noise level and sample size.
    
    Args:
        df: DataFrame containing the results
        metric: Name of the metric column to plot
        formulation: 'strong' or 'weak'
        output_dir: Directory to save the figure
        title_suffix: Additional text to append to the title
    """
    # Filter data for the specified formulation
    data = df[df['formulation'] == formulation]
    
    # Create pivot table for heatmap
    pivot = data.pivot(index='noise_level', columns='n_samples', values=metric)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': metric.upper().replace('_', ' ')})
    
    plt.title(f'{metric.upper().replace("_", " ")} - {formulation.capitalize()} Formulation{title_suffix}')
    plt.xlabel('Number of Samples')
    plt.ylabel('Noise Level')
    plt.tight_layout()
    
    # Save figure
    filename = f'heatmap_{metric}_{formulation}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def create_comparison_heatmap(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """
    Create a heatmap showing the ratio of weak/strong formulation performance.
    
    Args:
        df: DataFrame containing the results
        metric: Name of the metric column to compare
        output_dir: Directory to save the figure
    """
    # Separate strong and weak data
    strong = df[df['formulation'] == 'strong'].set_index(['noise_level', 'n_samples'])[metric]
    weak = df[df['formulation'] == 'weak'].set_index(['noise_level', 'n_samples'])[metric]
    
    # Calculate ratio (weak/strong)
    ratio = weak / strong
    ratio_pivot = ratio.reset_index().pivot(index='noise_level', columns='n_samples', values=metric)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    sns.heatmap(ratio_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', center=1.0,
                cbar_kws={'label': 'Ratio (Weak/Strong)'})
    
    plt.title(f'{metric.upper().replace("_", " ")} Ratio: Weak/Strong\n(Values < 1 indicate weak formulation is better)')
    plt.xlabel('Number of Samples')
    plt.ylabel('Noise Level')
    plt.tight_layout()
    
    # Save figure
    filename = f'heatmap_ratio_{metric}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def create_trend_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create line plots showing trends across noise levels and sample sizes.
    
    Args:
        df: DataFrame containing the results
        output_dir: Directory to save the figures
    """
    # RMSE vs Noise Level (averaged over sample sizes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RMSE vs Noise Level
    ax = axes[0]
    for formulation in ['strong', 'weak']:
        data = df[df['formulation'] == formulation].groupby('noise_level')['rmse'].agg(['mean', 'std'])
        ax.errorbar(data.index, data['mean'], yerr=data['std'], 
                   marker='o', label=formulation.capitalize(), capsize=5, linewidth=2)
    
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('RMSE (mean ± std)', fontsize=12)
    ax.set_title('RMSE vs Noise Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RMSE vs Sample Size
    ax = axes[1]
    for formulation in ['strong', 'weak']:
        data = df[df['formulation'] == formulation].groupby('n_samples')['rmse'].agg(['mean', 'std'])
        ax.errorbar(data.index, data['mean'], yerr=data['std'], 
                   marker='s', label=formulation.capitalize(), capsize=5, linewidth=2)
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('RMSE (mean ± std)', fontsize=12)
    ax.set_title('RMSE vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'trends_rmse.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def create_computation_time_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create visualizations for computation time analysis.
    
    Args:
        df: DataFrame containing the results
        output_dir: Directory to save the figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Computation time heatmap for weak formulation
    ax = axes[0]
    data = df[df['formulation'] == 'weak']
    pivot = data.pivot(index='noise_level', columns='n_samples', values='computation_time')
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax,
                cbar_kws={'label': 'Time (seconds)'})
    ax.set_title('Computation Time - Weak Formulation', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Noise Level')
    
    # Plot 2: Comparison of computation times (only for matching combinations)
    ax = axes[1]
    
    # Get strong and weak times with matching indices
    strong_data = df[df['formulation'] == 'strong'].set_index(['noise_level', 'n_samples'])['computation_time']
    weak_data = df[df['formulation'] == 'weak'].set_index(['noise_level', 'n_samples'])['computation_time']
    
    # Find common indices
    common_idx = strong_data.index.intersection(weak_data.index)
    
    if len(common_idx) > 0:
        strong_times = strong_data.loc[common_idx]
        weak_times = weak_data.loc[common_idx]
        
        ax.scatter(strong_times, weak_times, alpha=0.6, s=100)
        ax.plot([strong_times.min(), strong_times.max()], 
                [strong_times.min(), strong_times.max()], 
                'r--', label='Equal time', linewidth=2)
        ax.set_xlabel('Strong Formulation Time (s)', fontsize=12)
        ax.set_ylabel('Weak Formulation Time (s)', fontsize=12)
        ax.set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No matching parameter combinations\nfor comparison', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    plt.tight_layout()
    filename = 'computation_time_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def create_per_variable_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create visualizations comparing RMSE for each variable (x, y, z).
    
    Args:
        df: DataFrame containing the results
        output_dir: Directory to save the figures
    """
    variables = ['x', 'y', 'z']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        metric = f'rmse_{var}'
        
        # Group by noise level and formulation
        for formulation in ['strong', 'weak']:
            data = df[df['formulation'] == formulation].groupby('noise_level')[metric].agg(['mean', 'std'])
            ax.errorbar(data.index, data['mean'], yerr=data['std'], 
                       marker='o', label=formulation.capitalize(), capsize=5, linewidth=2)
        
        ax.set_xlabel('Noise Level', fontsize=11)
        ax.set_ylabel(f'RMSE_{var.upper()} (mean ± std)', fontsize=11)
        ax.set_title(f'Variable {var.upper()}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Per-Variable RMSE Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = 'per_variable_rmse.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate and save summary statistics.
    
    Args:
        df: DataFrame containing the results
        output_dir: Directory to save the summary
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall statistics by formulation
    print("\nOverall RMSE Statistics by Formulation:")
    print(df.groupby('formulation')['rmse'].describe())
    
    # Best performance (lowest RMSE) for each formulation
    print("\nBest Performance (Lowest RMSE):")
    for formulation in ['strong', 'weak']:
        best = df[df['formulation'] == formulation].nsmallest(1, 'rmse')
        print(f"\n{formulation.capitalize()}:")
        print(f"  Noise Level: {best['noise_level'].values[0]}")
        print(f"  Samples: {best['n_samples'].values[0]}")
        print(f"  RMSE: {best['rmse'].values[0]:.6f}")
        print(f"  Computation Time: {best['computation_time'].values[0]:.2f}s")
    
    # Worst performance (highest RMSE) for each formulation
    print("\nWorst Performance (Highest RMSE):")
    for formulation in ['strong', 'weak']:
        worst = df[df['formulation'] == formulation].nlargest(1, 'rmse')
        print(f"\n{formulation.capitalize()}:")
        print(f"  Noise Level: {worst['noise_level'].values[0]}")
        print(f"  Samples: {worst['n_samples'].values[0]}")
        print(f"  RMSE: {worst['rmse'].values[0]:.6f}")
        print(f"  Computation Time: {worst['computation_time'].values[0]:.2f}s")
    
    # Average improvement of weak over strong
    strong_rmse = df[df['formulation'] == 'strong'].set_index(['noise_level', 'n_samples'])['rmse']
    weak_rmse = df[df['formulation'] == 'weak'].set_index(['noise_level', 'n_samples'])['rmse']
    improvement = ((strong_rmse - weak_rmse) / strong_rmse * 100).mean()
    print(f"\nAverage RMSE Improvement (Weak vs Strong): {improvement:.2f}%")
    
    # Save summary to file
    summary_file = output_dir / 'summary_statistics.txt'
    with open(summary_file, 'w') as f:
        f.write("PARAMETRIC STUDY SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total parameter combinations: {len(df) // 2}\n")
        f.write(f"Noise levels: {sorted(df['noise_level'].unique())}\n")
        f.write(f"Sample sizes: {sorted(df['n_samples'].unique())}\n\n")
        f.write("\nOverall RMSE Statistics:\n")
        f.write(df.groupby('formulation')['rmse'].describe().to_string())
        f.write(f"\n\nAverage RMSE Improvement (Weak vs Strong): {improvement:.2f}%\n")
    
    print(f"\nSummary statistics saved to: {summary_file}")
    print("="*80 + "\n")


def main():
    """Main analysis workflow."""
    # Set up paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results_cluster'
    output_dir = results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    print(f"Saving analysis to: {output_dir}\n")
    
    # Load all results
    df = load_all_results(results_dir)
    
    # Generate summary statistics
    generate_summary_statistics(df, output_dir)
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    # Create heatmaps for key metrics
    for formulation in ['strong', 'weak']:
        create_heatmap(df, 'rmse', formulation, output_dir)
        create_heatmap(df, 'mse', formulation, output_dir)
    
    # Create comparison heatmaps
    create_comparison_heatmap(df, 'rmse', output_dir)
    
    # Create trend plots
    create_trend_plots(df, output_dir)
    
    # Create computation time analysis
    create_computation_time_analysis(df, output_dir)
    
    # Create per-variable analysis
    create_per_variable_analysis(df, output_dir)
    
    print("-" * 80)
    print(f"\nAnalysis complete! All figures saved to: {output_dir}")
    print(f"Total figures generated: {len(list(output_dir.glob('*.png')))}")


if __name__ == '__main__':
    main()
