
import numpy as np
import warnings
from ddmo.solver.scikit_solver import ScikitSolver
from ddmo.utils.data_constructor import DataConstructor

# Filter warnings
warnings.filterwarnings("ignore")

class MockLibrary:
    def transform(self, x):
        return x

def test_scikit_solver_multi_output():
    print("Testing ScikitSolver Multi-Output...")
    
    n_samples = 50
    n_features = 5
    n_targets = 3  # Multiple outputs
    
    x_dot = np.random.rand(n_samples, n_targets)
    theta = np.random.rand(n_samples, n_features)
    
    dc = DataConstructor(
        x_train=np.random.rand(n_samples, n_targets),
        time_array=np.linspace(0, 1, n_samples),
        x_dot=x_dot,
        library_object=lambda p: [],
        x_0=[np.random.rand(n_targets)],
        x_raw=np.random.rand(n_samples, n_targets)
    )
    
    # Payload
    lhs_payload = x_dot
    rhs_payload = {
        "theta": theta,
        "opt_vars": [1.0],
        "library": MockLibrary(),
        "colmask": None
    }

    # Test 1: LassoCV with MultiOutput (should work now)
    print("Testing LassoCV with MultiOutput wrapper...")
    solver = ScikitSolver(dc, "LassoCV", {"cv": 2})
    
    try:
        x_sim, coeffs, _, _, _ = solver.solve(lhs_payload, rhs_payload)
        
        if coeffs is not None:
             print("LassoCV solve completed.")
             print("Coefficients shape:", coeffs.shape)
             if coeffs.shape == (n_features, n_targets):
                 print("SUCCESS: LassoCV wrapped correctly.")
             else:
                 print(f"FAILURE: Shape mismatch. Expected {(n_features, n_targets)}, got {coeffs.shape}")
        else:
             print("FAILURE: Solve returned None.")
    except Exception as e:
        print(f"FAILURE Exception: {e}")

    # Test 2: MultiTaskLassoCV
    print("\nTesting MultiTaskLassoCV...")
    solver_mt = ScikitSolver(dc, "MultiTaskLassoCV", {"cv": 2})
    
    try:
        x_sim_mt, coeffs_mt, _, _, _ = solver_mt.solve(lhs_payload, rhs_payload)
        
        if coeffs_mt is not None:
             print("MultiTaskLassoCV solve completed.")
             print("Coefficients shape:", coeffs_mt.shape)
             if coeffs_mt.shape == (n_features, n_targets):
                 print("SUCCESS: MultiTaskLassoCV worked.")
             else:
                 print(f"FAILURE: Shape mismatch. Expected {(n_features, n_targets)}, got {coeffs_mt.shape}")
        else:
             print("FAILURE: Solve returned None.")
    except Exception as e:
        print(f"FAILURE Exception: {e}")

if __name__ == "__main__":
    test_scikit_solver_multi_output()
