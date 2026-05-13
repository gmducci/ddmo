
import warnings
import matplotlib.pyplot as plt
import numpy as np

import ddmo.differentiation.differentiation as diff
from scipy.integrate import odeint

from ddmo import identify
from mpl_toolkits.mplot3d import Axes3D
from ddmo.equations.goodwin_eqs import GoodwinModel
from ddmo.library_functions import LibraryBuilder
from ddmo import DataConstructor, FormulationSettings, OptimizerSettings, SolverSettings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # 1. Generate Data using Goodwin dynamics
    print("Generating synthetic data using Goodwin Model...")
    # Using default parameters which are set to produce oscillations
    # k1=1, k2=0.2, k3=1, k4=0.15, k5=1, k6=0.1, n=9.5
    goodwin = GoodwinModel(k1 = 1, 
                           k2 = 0.2, 
                           k3 = 1, 
                           k4 = 0.15, 
                           k5 = 1, 
                           k6 = 0.1, 
                           n = 9.5)
    
    
    
    # Time vector
    t_start, t_end, dt = 0, 250, 0.05
    t = np.arange(t_start, t_end, dt)
    
    # Initial conditions [x, y, z]
    x0 = [0., 1, 1.]
    
    # Integrate dynamics
    x_clean = odeint(goodwin.dynamics, x0, t)
    sigma_per_var = np.std(x_clean, axis=0) 

    # Define noise level as a percentage of that variation (e.g., 5%)
    noise_ratio = 0.15

    # Generate noise scaled to each variable's std dev
    noise = np.random.normal(0, 1, x_clean.shape) * sigma_per_var * noise_ratio
    x_noisy = x_clean + noise
    
    # Estimate derivatives using pynumdiff Kalman method
    print("Estimating derivatives with pynumdiff (Kalman)...")
    x_dot_estimated = np.zeros_like(x_noisy)
    x_hat_estimated = np.zeros_like(x_noisy)
    
    MyDerivatives = diff.Derivatives("kalman", t, optimization = False)
    x_hat0, x_dot_hat0 = MyDerivatives.get_derivatives(x_noisy)
    x_hat, x_dot_hat = MyDerivatives.get_derivatives(x_hat0)
    
    # 2. Define the Library
    # The Goodwin model contains linear terms and a Hill function term: 1/(1+z^n)
    # add_hill adds 1/(1+x^p) type terms.
    library = (
        LibraryBuilder()
        .add_polynomial(max_degree=2, include_interactions=True)
        .add_hill(exponent_param="n_hill")  # 1/(1+x^n)
        .add_exponential(rate_param="gamma_2")  # exp(-γ₂·x)
    )

    # 3. Construct Data Container
    data = DataConstructor(
        x_train=x_noisy[20:-20, :], # Or use x_hat_estimated if preferred, but usually we train on noisy data (or smoothed?)
        # User asked for x_dot estimates, usually implies we might want to use strong formulation or at least provide them.
        x_dot =x_dot_hat[20:-20, :],
        time_array=t[20:-20],
        library_object=library.build,
        x_0=[x0],
        x_raw=x_noisy[20:-20, :]
    )

    # 4. Settings
    # Optimizer settings 
    # The 'n' parameter in Goodwin is around 9.5
    library_bounds = {
        "n_hill": (2.0, 10.0),
        "gamma_2": (0.0, 1.0),
    }

    # Weak formulation optimization parameters
    formulation_bounds = {
        # "window_length": (21, 101),
        # "stride": (1, 5),
    }

    # Solver bounds: Optimize the SINDy hyperparameters
    solver_bounds = {
        "threshold": (.001, 2),
        "alpha": (0.001, 1)
    }

    optimizer_settings = OptimizerSettings(
        library_bounds=library_bounds,
        formulation_bounds=formulation_bounds,
        solver_bounds=solver_bounds,
        n_obj=2,
        n_gen=10, 
        pop_size=100,
        algorithm="NSGA3"
    )

    # Solver settings
    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "normalize_columns": False,
            # "threshold": 0.5, 
            # "nu": 10,
            # "fit_intercept": False,
            # "alpha": 0.001,
            # "thresholder": 'L0',
        },
        scale_features=True,
        ensemble={
            "enabled": False,
            "bagging": True,
            "n_models": 20,
            "n_candidates_to_drop": 3
        }
    )

    # Formulation settings -> WEAK
    # We can also start with an initial guess for parameters if not optimized,
    # but since we provided bounds in optimizer_settings, they will be optimized.
    formulation_settings = FormulationSettings(
        formulation="strong",
        # test_function="PolynomialWindow",
        # # # Default values if optimization doesn't override them immediately or for initial population
        # window_length=21, 
        # stride=1,
        # degree=2
    )

    # 5. Run Identification
    print("Running identification...")
    result = identify(
        data_instance=data,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings
    )

    # 6. Visualize Results
    print("Identification complete.")
    
    if hasattr(result, 'solutions') and len(result.solutions) > 0:
        trajectory_pred = result.solutions[0]
        
        plt.figure(figsize=(12, 10))
        variable_names = ['x', 'y', 'z']
        
        for i, name in enumerate(variable_names):
            plt.subplot(3, 1, i + 1)
            plt.plot(t, x_clean[:, i], 'k-', linewidth=2, label=f'True {name}')
            plt.plot(t, x_noisy[:, i], 'k.', alpha=0.3, label=f'Noisy {name}')
            if trajectory_pred is not None:
                # Need to handle case where trajectory might be shorter or different shape depending on how it's returned
                # Typically result.solutions[0] matches x_train shape (possibly cropped by window/integration?)
                # Usually identify returns the integrated trajectory on the same time inputs if possible.
                
                # Check shape compatibility
                if len(trajectory_pred) == len(t):
                    plt.plot(t, trajectory_pred[:, i], 'r--', linewidth=2, label=f'Identified {name}')
                else:
                     # Fallback if specific integration range was returned
                    plt.plot(t[:len(trajectory_pred)], trajectory_pred[:, i], 'r--', linewidth=2, label=f'Identified {name}')
            
            plt.ylabel(name)
            plt.legend()
            if i == 0:
                plt.title("Goodwin Model Identification (Weak Formulation)")
                
        plt.xlabel("Time")
        plt.tight_layout()

        # Phase Space Visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot True Dynamics
        ax.plot(x_clean[:, 0], x_clean[:, 1], x_clean[:, 2], 'k-', linewidth=1.5, label='True Dynamics')
        
        # Plot Noisy Data
        ax.scatter(x_noisy[::2, 0], x_noisy[::2, 1], x_noisy[::2, 2], c='gray', s=10, alpha=0.3, label='Noisy Data')
        
        # Plot Identified Dynamics
        if trajectory_pred is not None:
            # Handle potential length mismatch safely
            L = min(len(trajectory_pred), len(t))
            ax.plot(trajectory_pred[:L, 0], trajectory_pred[:L, 1], trajectory_pred[:L, 2], 'r--', linewidth=2, label='Identified Dynamics')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Goodwin Model Phase Space')
        ax.legend()
        plt.show()
        
        # Print coefficients
        print("\nIdentified Coefficients (first Pareto front solution):")
        # Accessing coefficients depends on exact return structure of result
        # Assuming result has a way to inspect the model or we look at printed output from 'identify' usually.
        if hasattr(result, 'models') and len(result.models) > 0:
             result.models[0].print()

    else:
        print("No solutions found.")
