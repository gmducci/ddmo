
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from ddmo import identify
from ddmo.equations.brussellator import Brussellator
from ddmo.library_functions import LibraryBuilder
from ddmo.solver.settings import SolverSettings
from ddmo.utils.constructors import OptimizerSettings
from ddmo.utils.data_constructor import DataConstructor
from ddmo.formulation.formulation_settings import FormulationSettings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # 1. Generate Data using Brussellator dynamics
    print("Generating synthetic data...")
    brussellator = Brussellator(a=1, b=3, alpha=2.5, beta=2)
    
    # Time vector
    t_start, t_end, dt = 0, 40, 0.05
    t = np.arange(t_start, t_end, dt)
    
    # Initial conditions
    x0 = [1.0, 1.0]
    
    # Integrate dynamics
    x_clean = odeint(brussellator.dynamics, x0, t)
    
    # Add Gaussian noise
    # Weak formulation is robust to noise, so we can add some visible noise
    noise_level = 0.05
    x_noisy = x_clean + np.random.normal(0, noise_level, x_clean.shape)
    
    # For data constructor, we pass the noisy data. 
    # x_dot is optional/unused for weak formulation usually, but good to provide if available or just None.
    # We will pass None and let the formulation handle derivatives internally if needed (weak form avoids direct derivative computation).
    
    # 2. Define the Library
    # Brussellator involves terms like x, x^2*y. This requires polynomial degree 3.
    library = (
        LibraryBuilder()
        .add_polynomial(max_degree=3, include_interactions=True)
        .add_custom(lambda x, y, p: (x ** p["g1"]) * (y ** p["g2"]),
            lambda x, y, p: f"x^{p['g1']:.2f}*y^{p['g2']:.2f}")
    )

    # 3. Construct Data Container
    data = DataConstructor(
        x_train=x_noisy,
        time_array=t,
        x_dot=None, # Weak formulation does not strictly need pre-computed derivatives of noisy data
        library_object=library.build,
        x_0=[x0],
        x_raw=x_noisy 
    )

    # 4. Settings
    # Optimizer settings (Genetic Algorithm)
    # We MUST provide at least one variable to optimize to avoid pymoo crashing.
    # For Weak Formulation, 'window_length' is a key parameter to optimize.
    formulation_bounds = {
        "window_length": (81, 201), # Must be odd; formulation handles conversion usually or we step carefully
        "stride": (5, 70),
        "degree": (1, 5)
    }

    optimizer_settings = OptimizerSettings(
        library_bounds={"g1": (0, 5), "g2": (0, 5)},
        formulation_bounds=formulation_bounds,
        n_obj=2,  # Multi-objective: Complexity vs Error
        n_gen=15, 
        pop_size=150,
        algorithm="NSGA3"
    )

    # Solver settings (PySINDy / STLSQ)
    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={
            "fit_intercept": True,
            "alpha": 0.1,
            "threshold": 0.25
        },
        scale_features=True
    )

    # Formulation settings -> WEAK
    formulation_settings = FormulationSettings(
        formulation="weak",
        test_function="PolynomialWindow",
        degree=4,         # Can be kept fixed or optimized
        window_length=51, # specific values if not optimizing, or initial guesses
        stride=5
    )

    # 5. Run Identification
    print("Running identification with Weak Formulation...")
    result = identify(
        data_instance=data,
        optimizer_settings=optimizer_settings,
        solver_settings=solver_settings,
        formulation_settings=formulation_settings
    )

    # 6. Visualize Results
    print("Identification complete.")
    
    if len(result.solutions) > 0:
        trajectory_pred = result.solutions[0]
        
        plt.figure(figsize=(12, 6))
        
        for i, name in enumerate(['x', 'y']):
            plt.subplot(2, 1, i + 1)
            plt.plot(t, x_clean[:, i], 'k-', linewidth=2, label=f'True {name}')
            plt.plot(t, x_noisy[:, i], 'k.', alpha=0.3, label=f'Noisy {name}')
            if trajectory_pred is not None:
                plt.plot(t, trajectory_pred[:, i], 'r--', linewidth=2, label=f'Identified {name}')
            plt.ylabel(name)
            plt.legend()
            if i == 0:
                plt.title("Brussellator Identification (Weak Formulation)")
                
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
    else:
        print("No solutions found.")
