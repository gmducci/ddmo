
import numpy as np
import warnings
from ddmo.solver.scikit_solver import ScikitSolver
from ddmo.utils.data_constructor import DataConstructor

warnings.filterwarnings("ignore")

class MockLibrary:
    def transform(self, x):
        return x

def test_scikit_solver_lasso_ridge():
    print("Testing ScikitSolver Lasso and Ridge...")
    
    n_samples = 50
    n_features = 5
    n_targets = 2
    
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
    
    lhs_payload = x_dot
    rhs_payload = {
        "theta": theta,
        "opt_vars": [1.0],
        "library": MockLibrary(),
        "colmask": None
    }

    # Test Lasso
    print("Testing Lasso...")
    solver_lasso = ScikitSolver(dc, "Lasso", {"alpha": 0.1})
    try:
        _, coeffs, _, _, _ = solver_lasso.solve(lhs_payload, rhs_payload)
        print("Lasso Coefficients shape:", coeffs.shape)
    except Exception as e:
        print(f"Lasso Failed: {e}")

    # Test Ridge
    print("\nTesting Ridge...")
    solver_ridge = ScikitSolver(dc, "Ridge", {"alpha": 1.0})
    try:
        _, coeffs_r, _, _, _ = solver_ridge.solve(lhs_payload, rhs_payload)
        print("Ridge Coefficients shape:", coeffs_r.shape)
    except Exception as e:
        print(f"Ridge Failed: {e}")

if __name__ == "__main__":
    test_scikit_solver_lasso_ridge()
