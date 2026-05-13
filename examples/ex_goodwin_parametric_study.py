#!/usr/bin/env python3
"""
Parametric Study: Goodwin Model Reconstruction
Strong vs Weak Formulation Comparison

This script performs a 2D parametric study varying:
- Noise level
- Number of samples

For each grid point, both strong and weak formulations are evaluated
on identical data with the same random seed for reproducibility.
"""

import argparse
import warnings
import json
import time
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint

import ddmo.differentiation.differentiation as diff
from ddmo import identify
from ddmo.equations.goodwin_eqs import GoodwinModel
from ddmo.library_functions import LibraryBuilder
from ddmo.solver.settings import SolverSettings
from ddmo.utils.constructors import OptimizerSettings
from ddmo.utils.data_constructor import DataConstructor
from ddmo.formulation.formulation_settings import FormulationSettings

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments for parametric study."""
    parser = argparse.ArgumentParser(
        description='Parametric study of Goodwin model reconstruction'
    )
    
    # Job array support (main argument for cluster)
    parser.add_argument('--job-id', type=int, default=None,
                        help='Job array index (if running on cluster). If not specified, runs all grid points.')
    parser.add_argument('--total-jobs', type=int, default=None,
                        help='Total number of jobs in the array. If specified, grid points are distributed across jobs.')
    
    # Optimization settings
    parser.add_argument('--n-gen', type=int, default=10,
                        help='Number of generations for optimization (default: 10)')
    parser.add_argument('--pop-size', type=int, default=100,
                        help='Population size for optimization (default: 100)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='./results_parametric',
                        help='Output directory for results (default: ./results_parametric)')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base random seed (default: 42)')
    
    # Formulation selection
    parser.add_argument('--formulations', type=str, nargs='+',
                        default=['strong', 'weak'],
                        choices=['strong', 'weak'],
                        help='Formulations to test (default: strong weak)')
    
    return parser.parse_args()


def define_parameter_grid():
    """
    Define the 2D parameter grid for the parametric study.
    
    Modify this function to change the parameter space.
    
    Returns
    -------
    dict
        Dictionary with 'noise_levels' and 'n_samples' lists
    """
    # Define your parameter grid here
    noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
    # Extended sample range up to 15000
    n_samples = [2500, 5000, 7500, 10000, 12500, 15000]
    
    return {
        'noise_levels': noise_levels,
        'n_samples': n_samples
    }


def generate_goodwin_data(n_samples, noise_level, random_seed):
    """
    Generate Goodwin model data with specified noise level.
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    noise_level : float
        Noise level as fraction of signal std
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing time array, clean data, noisy data, 
        derivative estimates, and metadata
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Initialize Goodwin model with standard parameters
    goodwin = GoodwinModel(
        k1=1.0, k2=0.2, k3=1.0, 
        k4=0.15, k5=1.0, k6=0.1, 
        n=9.5
    )
    
    # Time vector - adjust t_end to get desired n_samples
    dt = 0.05
    t_end = n_samples * dt
    t = np.arange(0, t_end, dt)
    
    # Initial conditions [x, y, z]
    x0 = [0., 1., 1.]
    
    # Integrate clean dynamics
    x_clean = odeint(goodwin.dynamics, x0, t)
    
    # Add noise
    sigma_per_var = np.std(x_clean, axis=0)
    noise = np.random.normal(0, 1, x_clean.shape) * sigma_per_var * noise_level
    x_noisy = x_clean + noise
    
    # Estimate derivatives using Kalman smoother
    MyDerivatives = diff.Derivatives("kalman", t, optimization=False)
    x_hat0, x_dot_hat0 = MyDerivatives.get_derivatives(x_noisy)
    x_hat, x_dot_hat = MyDerivatives.get_derivatives(x_hat0)
    
    return {
        'time': t,
        'x_clean': x_clean,
        'x_noisy': x_noisy,
        'x_hat': x_hat,
        'x_dot_hat': x_dot_hat,
        'x0': x0,
        'noise_level': noise_level,
        'n_samples': len(t),
        'random_seed': random_seed,
        'goodwin_params': {
            'k1': 1.0, 'k2': 0.2, 'k3': 1.0,
            'k4': 0.15, 'k5': 1.0, 'k6': 0.1, 'n': 9.5
        }
    }


def create_library():
    """Create the library of candidate functions."""
    return (
        LibraryBuilder()
        .add_polynomial(max_degree=2, include_interactions=True)
        .add_hill(exponent_param="n_hill")
        .add_exponential(rate_param="gamma_2")
    )


def run_identification(data_dict, formulation_type, n_gen, pop_size):
    """
    Run identification for a specific formulation.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary from generate_goodwin_data
    formulation_type : str
        'strong' or 'weak'
    n_gen : int
        Number of generations
    pop_size : int
        Population size
        
    Returns
    -------
    dict
        Results dictionary containing model, metrics, and timing
    """
    start_time = time.time()
    
    # Create library
    library = create_library()
    
    # Trim edges only for strong formulation (weak formulation handles boundaries via windowing)
    if formulation_type == 'strong':
        trim = 20
        t = data_dict['time'][trim:-trim]
        x_noisy = data_dict['x_noisy'][trim:-trim, :]
        x_dot_hat = data_dict['x_dot_hat'][trim:-trim, :]
    else:
        # Weak formulation: use full dataset
        t = data_dict['time']
        x_noisy = data_dict['x_noisy']
        x_dot_hat = data_dict['x_dot_hat']
    
    x0 = data_dict['x0']
    
    # Construct data container
    data = DataConstructor(
        x_train=x_noisy,
        x_dot=x_dot_hat,
        time_array=t,
        library_object=library.build,
        x_0=[x0],
        x_raw=x_noisy
    )
    
    # Library bounds (same for both formulations)
    library_bounds = {
        "n_hill": (2.0, 10.0),
        "gamma_2": (0.0, 1.0),
    }
    
    # Solver bounds (same for both formulations)
    solver_bounds = {
        "threshold": (0.001, 2.0),
        "alpha": (0.001, 1.0)
    }
    
    # Formulation-specific bounds
    if formulation_type == 'weak':
        formulation_bounds = {
            "window_length": (21, 101),
            "stride": (1, 5),
        }
    else:
        formulation_bounds = {}
    
    # Optimizer settings
    optimizer_settings = OptimizerSettings(
        library_bounds=library_bounds,
        formulation_bounds=formulation_bounds,
        solver_bounds=solver_bounds,
        n_obj=2,
        n_gen=n_gen,
        pop_size=pop_size,
        algorithm="NSGA3"
    )
    
    # Solver settings
    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "normalize_columns": False,
        },
        scale_features=True,
        ensemble={
            "enabled": False,
            "bagging": True,
            "n_models": 20,
            "n_candidates_to_drop": 3
        }
    )
    
    # Formulation settings
    if formulation_type == 'weak':
        formulation_settings = FormulationSettings(
            formulation="weak",
            test_function="PolynomialWindow",
            window_length=51,  # Default value (will be optimized)
            stride=2,          # Default value (will be optimized)
            degree=2
        )
    else:
        formulation_settings = FormulationSettings(
            formulation="strong"
        )
    
    # Run identification with error handling
    try:
        print(f"Starting optimization for {formulation_type} formulation...")
        result = identify(
            data_instance=data,
            optimizer_settings=optimizer_settings,
            solver_settings=solver_settings,
            formulation_settings=formulation_settings
        )
        print(f"Optimization completed successfully!")
    except Exception as e:
        print(f"ERROR during identification: {e}")
        import traceback
        traceback.print_exc()
        # Return error result
        elapsed_time = time.time() - start_time
        return {
            'formulation': formulation_type,
            'result_object': None,
            'computation_time': elapsed_time,
            'n_gen': n_gen,
            'pop_size': pop_size,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }
    
    elapsed_time = time.time() - start_time
    
    # Package results
    results = {
        'formulation': formulation_type,
        'result_object': result,
        'computation_time': elapsed_time,
        'n_gen': n_gen,
        'pop_size': pop_size,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def compute_metrics(result, data_dict):
    """
    Compute performance metrics for a result.
    
    Parameters
    ----------
    result : dict
        Results from run_identification
    data_dict : dict
        Original data dictionary
        
    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    metrics = {}
    
    result_obj = result['result_object']
    formulation_type = result['formulation']
    
    if hasattr(result_obj, 'solutions') and len(result_obj.solutions) > 0:
        # Get first Pareto solution (or implement selection strategy)
        trajectory_pred = result_obj.solutions[0]
        
        # Trim to match prediction length (only for strong formulation)
        if formulation_type == 'strong':
            trim = 20
            x_clean = data_dict['x_clean'][trim:-trim, :]
            x_noisy = data_dict['x_noisy'][trim:-trim, :]
        else:
            # Weak formulation uses full dataset
            x_clean = data_dict['x_clean']
            x_noisy = data_dict['x_noisy']
        
        # Compute trajectory errors
        if trajectory_pred is not None:
            min_len = min(len(trajectory_pred), len(x_clean))
            
            # MSE and RMSE
            mse = np.mean((trajectory_pred[:min_len] - x_clean[:min_len])**2)
            rmse = np.sqrt(mse)
            
            # Per-variable RMSE
            rmse_per_var = np.sqrt(np.mean((trajectory_pred[:min_len] - x_clean[:min_len])**2, axis=0))
            
            metrics['mse'] = float(mse)
            metrics['rmse'] = float(rmse)
            metrics['rmse_x'] = float(rmse_per_var[0])
            metrics['rmse_y'] = float(rmse_per_var[1])
            metrics['rmse_z'] = float(rmse_per_var[2])
            
            # Normalized error (relative to signal magnitude)
            signal_range = np.ptp(x_clean[:min_len], axis=0)
            normalized_rmse = rmse_per_var / signal_range
            metrics['normalized_rmse'] = normalized_rmse.tolist()
        
        # Model complexity (number of non-zero terms)
        if hasattr(result_obj, 'models') and len(result_obj.models) > 0:
            model = result_obj.models[0]
            # This depends on your model structure
            # metrics['n_terms'] = count_nonzero_terms(model)
            
    else:
        metrics['error'] = 'No solutions found'
    
    return metrics


def save_results(grid_point_results, output_dir, noise_level, n_samples, data_dict):
    """
    Save results for a single grid point.
    
    Saves:
    - metrics.json: Summary metrics for both formulations
    - metadata.csv: Grid point metadata
    - trajectories_*.npy: Predicted trajectories (if available)
    - data_noisy.npy: Noisy training data
    - data_clean.npy: Clean reference data
    - pymoo_result_*.pkl: Complete pymoo result objects (pickled)
    
    Parameters
    ----------
    grid_point_results : dict
        Results for this grid point
    output_dir : Path
        Output directory
    noise_level : float
        Noise level for this grid point
    n_samples : int
        Number of samples for this grid point
    data_dict : dict
        Data dictionary containing noisy and clean data
    """
    # Create subdirectory for this grid point
    subdir = output_dir / f"noise_{noise_level:.3f}_samples_{n_samples}"
    subdir.mkdir(parents=True, exist_ok=True)
    
    # Save noisy and clean data
    np.save(subdir / "data_noisy.npy", data_dict['x_noisy'])
    np.save(subdir / "data_clean.npy", data_dict['x_clean'])
    np.save(subdir / "time_array.npy", data_dict['time'])
    
    # Save metadata as CSV
    metadata = {
        'noise_level': noise_level,
        'n_samples': n_samples,
        'grid_index': grid_point_results.get('grid_index', -1),
        'random_seed': grid_point_results.get('random_seed', -1),
        'actual_samples': grid_point_results.get('data_metadata', {}).get('actual_samples', n_samples)
    }
    
    import pandas as pd
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(subdir / "metadata.csv", index=False)
    
    # Save metrics summary (JSON)
    metrics_summary = {
        'noise_level': noise_level,
        'n_samples': n_samples,
        'grid_index': grid_point_results.get('grid_index', -1),
        'formulations': {}
    }
    
    # Process each formulation
    for formulation in grid_point_results['formulations']:
        form_type = formulation['formulation']
        
        # Save metrics
        metrics_summary['formulations'][form_type] = {
            'metrics': formulation.get('metrics', {}),
            'computation_time': formulation.get('computation_time', 0),
            'n_gen': formulation.get('n_gen'),
            'pop_size': formulation.get('pop_size'),
            'timestamp': formulation.get('timestamp')
        }
        
        if 'error' in formulation:
            metrics_summary['formulations'][form_type]['error'] = formulation['error']
        
        # Save trajectory if available
        if 'result_object' in formulation and formulation['result_object'] is not None:
            result_obj = formulation['result_object']
            if hasattr(result_obj, 'solutions') and len(result_obj.solutions) > 0:
                trajectory = result_obj.solutions[0]
                if trajectory is not None:
                    np.save(subdir / f"trajectory_{form_type}.npy", trajectory)
            
            # Extract and save structured data from result object
            try:
                # Save all trajectories (all Pareto solutions)
                if hasattr(result_obj, 'solutions') and result_obj.solutions:
                    all_trajectories = np.array(result_obj.solutions)
                    np.save(subdir / f"all_trajectories_{form_type}.npy", all_trajectories)
                    print(f"  Saved {len(result_obj.solutions)} trajectories for {form_type}")
                
                # Save Pareto front objectives (F values)
                if hasattr(result_obj, 'F') and result_obj.F is not None:
                    np.save(subdir / f"pareto_objectives_{form_type}.npy", result_obj.F)
                    # Also save as CSV for easy viewing
                    objectives_df = pd.DataFrame(result_obj.F, 
                                                 columns=[f'objective_{i}' for i in range(result_obj.F.shape[1])])
                    objectives_df.to_csv(subdir / f"pareto_objectives_{form_type}.csv", index=False)
                    print(f"  Saved Pareto front objectives ({result_obj.F.shape[0]} solutions)")
                
                # Save decision variables (X values - optimization parameters)
                if hasattr(result_obj, 'X') and result_obj.X is not None:
                    np.save(subdir / f"decision_variables_{form_type}.npy", result_obj.X)
                    # Create meaningful column names based on what was optimized
                    param_names = []
                    if hasattr(result_obj, 'parameter_names'):
                        param_names = result_obj.parameter_names
                    else:
                        param_names = [f'param_{i}' for i in range(result_obj.X.shape[1])]
                    
                    decision_df = pd.DataFrame(result_obj.X, columns=param_names)
                    decision_df.to_csv(subdir / f"decision_variables_{form_type}.csv", index=False)
                    print(f"  Saved decision variables ({result_obj.X.shape[0]} solutions)")
                
                # Save model coefficients if available
                if hasattr(result_obj, 'models') and result_obj.models:
                    for idx, model in enumerate(result_obj.models):
                        if hasattr(model, 'coefficients'):
                            coef_array = np.array(model.coefficients())
                            np.save(subdir / f"model_coefficients_{form_type}_sol{idx}.npy", coef_array)
                        
                        # Save feature names if available
                        if hasattr(model, 'get_feature_names'):
                            feature_names = model.get_feature_names()
                            with open(subdir / f"feature_names_{form_type}_sol{idx}.txt", 'w') as f:
                                f.write('\n'.join(feature_names))
                    
                    print(f"  Saved coefficients for {len(result_obj.models)} models")
                
                # Save optimization history if available
                if hasattr(result_obj, 'history') and result_obj.history:
                    history_data = []
                    for gen_idx, gen_data in enumerate(result_obj.history):
                        if hasattr(gen_data, 'opt') and hasattr(gen_data.opt, 'F'):
                            # Save best objectives per generation
                            best_f = gen_data.opt.F
                            if len(best_f.shape) == 1:
                                best_f = best_f.reshape(1, -1)
                            for sol_idx, obj_vals in enumerate(best_f):
                                row = {'generation': gen_idx, 'solution_idx': sol_idx}
                                for obj_idx, val in enumerate(obj_vals):
                                    row[f'objective_{obj_idx}'] = val
                                history_data.append(row)
                    
                    if history_data:
                        history_df = pd.DataFrame(history_data)
                        history_df.to_csv(subdir / f"optimization_history_{form_type}.csv", index=False)
                        print(f"  Saved optimization history ({len(result_obj.history)} generations)")
                
            except Exception as e:
                print(f"  Warning: Could not extract some result data for {form_type}: {e}")
            
            # Save complete pymoo result object (pickled) as backup
            import pickle
            try:
                with open(subdir / f"pymoo_result_{form_type}.pkl", 'wb') as f:
                    pickle.dump(result_obj, f)
                print(f"  Saved complete pymoo result object (pickle) for {form_type}")
            except Exception as e:
                print(f"  Warning: Could not pickle pymoo result for {form_type}: {e}")
    
    # Save JSON summary
    with open(subdir / "metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Create a combined metrics CSV for easy analysis
    metrics_rows = []
    for formulation in grid_point_results['formulations']:
        form_type = formulation['formulation']
        metrics = formulation.get('metrics', {})
        
        row = {
            'formulation': form_type,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'computation_time': formulation.get('computation_time', 0),
        }
        
        # Add all metrics
        row.update(metrics)
        metrics_rows.append(row)
    
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(subdir / "metrics.csv", index=False)
    
    print(f"Results saved to {subdir}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parameter grid from hardcoded definition
    param_grid = define_parameter_grid()
    noise_levels = param_grid['noise_levels']
    n_samples_list = param_grid['n_samples']
    
    # Generate all grid points (cartesian product)
    grid_points = [
        (noise, n_samp) 
        for noise in noise_levels 
        for n_samp in n_samples_list
    ]
    
    total_grid_points = len(grid_points)
    
    print(f"Parameter Grid Definition:")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Sample counts: {n_samples_list}")
    print(f"  Total grid points: {total_grid_points}")
    print()
    
    # Handle job array indexing if specified
    if args.job_id is not None:
        if args.total_jobs is not None:
            # Distribute grid points across jobs
            # Each job processes a chunk of grid points
            points_per_job = total_grid_points // args.total_jobs
            remainder = total_grid_points % args.total_jobs
            
            # Calculate start and end indices for this job
            if args.job_id < remainder:
                # First 'remainder' jobs get one extra point
                start_idx = args.job_id * (points_per_job + 1)
                end_idx = start_idx + points_per_job + 1
            else:
                # Remaining jobs get 'points_per_job' points
                start_idx = remainder * (points_per_job + 1) + (args.job_id - remainder) * points_per_job
                end_idx = start_idx + points_per_job
            
            # Select grid points for this job
            grid_points_for_job = grid_points[start_idx:end_idx]
            
            print(f"Running as job array task {args.job_id}/{args.total_jobs}")
            print(f"Processing grid points {start_idx} to {end_idx-1} ({len(grid_points_for_job)} points)")
            print(f"Grid points: {[(f'noise={n:.3f}, samples={s}') for n, s in grid_points_for_job]}")
            print()
            
            # Store indices for proper seed calculation
            grid_indices = list(range(start_idx, end_idx))
            grid_points = grid_points_for_job
            
        else:
            # Single grid point mode (original behavior)
            if args.job_id >= total_grid_points:
                print(f"ERROR: Job ID {args.job_id} exceeds grid size {total_grid_points}")
                print(f"Valid job IDs: 0 to {total_grid_points - 1}")
                return
            
            # Select only the grid point for this job
            grid_points = [grid_points[args.job_id]]
            grid_indices = [args.job_id]
            
            print(f"Running as job array task {args.job_id}/{total_grid_points}")
            print(f"Processing grid point: noise={grid_points[0][0]}, samples={grid_points[0][1]}")
            print()
    else:
        # Run all grid points sequentially
        grid_indices = list(range(total_grid_points))
        print(f"Running all {total_grid_points} grid points sequentially")
        print()
    
    # Process each grid point
    for idx, (noise_level, n_samples) in enumerate(grid_points):
        # Get the actual grid index (important for seed calculation)
        actual_idx = grid_indices[idx]
        
        print(f"\n{'='*60}")
        print(f"Grid point {actual_idx + 1}/{total_grid_points}")
        print(f"Noise level: {noise_level}, Samples: {n_samples}")
        print(f"{'='*60}\n")
        
        # Generate unique seed for this grid point
        grid_seed = args.seed_base + actual_idx
        
        # Generate data
        print("Generating data...")
        data_dict = generate_goodwin_data(n_samples, noise_level, grid_seed)
        
        # Store results for this grid point
        grid_point_results = {
            'noise_level': noise_level,
            'n_samples': n_samples,
            'grid_index': actual_idx,
            'random_seed': grid_seed,
            'data_metadata': {
                'actual_samples': data_dict['n_samples'],
                'goodwin_params': data_dict['goodwin_params']
            },
            'formulations': []
        }
        
        # Run both formulations on the same data
        for formulation_type in args.formulations:
            print(f"\n{'='*60}")
            print(f"Running {formulation_type} formulation...")
            print(f"{'='*60}")
            
            try:
                result = run_identification(
                    data_dict, 
                    formulation_type, 
                    args.n_gen, 
                    args.pop_size
                )
                
                print(f"Computation time: {result['computation_time']:.2f}s")
                
                # Compute metrics
                print("Computing metrics...")
                metrics = compute_metrics(result, data_dict)
                result['metrics'] = metrics
                
                print(f"Metrics: {metrics}")
                
                grid_point_results['formulations'].append(result)
                print(f"✓ {formulation_type.capitalize()} formulation completed successfully")
                
            except Exception as e:
                print(f"✗ ERROR in {formulation_type} formulation: {e}")
                import traceback
                traceback.print_exc()
                # Store error result
                grid_point_results['formulations'].append({
                    'formulation': formulation_type,
                    'error': str(e),
                    'computation_time': 0
                })
        
        # Save results for this grid point
        print(f"\n{'='*60}")
        print(f"Saving results for grid point {actual_idx}...")
        print(f"{'='*60}")
        try:
            save_results(grid_point_results, output_dir, noise_level, n_samples, data_dict)
            print(f"✓ Results saved successfully!")
        except Exception as e:
            print(f"✗ ERROR saving results: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Parametric study complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()