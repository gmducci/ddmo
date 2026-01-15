
import sys
from unittest.mock import MagicMock, patch

# Mock pymoo if it fails to import
try:
    from pymoo.core.problem import Problem
except ImportError:
    # Create a dummy Problem class if pymoo is completely broken
    class Problem:
        def __init__(self, *args, **kwargs):
            pass

# Mocking modules that might trigger lzma or other issues
sys.modules['pymoo.algorithms.moo.nsga2'] = MagicMock()
sys.modules['pymoo.algorithms.moo.nsga3'] = MagicMock()
sys.modules['pymoo.optimize'] = MagicMock()

from ddmo.utils.constructors import OptimizerSettings
from ddmo.utils.data_constructor import DataConstructor
from ddmo.solver.pysindy_solver import Pysindy_Solver
# We need to be careful importing Optimizer if it imports pymoo
from ddmo.optimizer.optimizer import Optimizer
import numpy as np

def test_optimizer_settings():
    print("Testing OptimizerSettings...")
    settings = OptimizerSettings(
        library_bounds={"a": (0, 1)},
        formulation_bounds={"b": (0, 1)},
        solver_bounds={"threshold": (0.01, 0.1)},
        n_obj=2,
        n_gen=10
    )
    assert settings.n_var == 3
    assert "threshold" in settings.solver_param_names
    assert settings.all_bounds["threshold"] == (0.01, 0.1)
    print("OptimizerSettings passed.")

def test_pysindy_solver_kwargs():
    print("Testing Pysindy_Solver kwargs...")
    data = MagicMock(spec=DataConstructor)
    solver = Pysindy_Solver(data, optimizer="STLSQ", solver_parameters={"alpha": 0.5})
    
    # Mock get_optimizer to return a class that we can inspect
    with patch("ddmo.solver.pysindy_solver.get_optimizer") as mock_get_opt:
        mock_cls = MagicMock()
        mock_get_opt.return_value = mock_cls
        
        # Test solve with kwargs
        # We need to simulate arguments passed to solve
        # solve calls builder.build_problem(args[0]) if 1 arg
        # or expects (lhs, rhs) if 2 args.
        
        # Let's try 2 args path which is simpler
        lhs = np.array([[1, 2]])
        rhs = {"opt_vars": [1], "theta": [[1]], "library": MagicMock()}
        
        # Mock SINDyOptimizer
        with patch("ddmo.solver.pysindy_solver.SINDyOptimizer") as mock_sindy:
             solver.solve(lhs, rhs, threshold=0.99)
             
             # Check if threshold was passed to base_optimizer init (via mock_cls)
             # The code does: base_optimizer = get_optimizer(...)(**params_dict)
             # params_dict should contain 'alpha': 0.5 and 'threshold': 0.99
             call_kwargs = mock_cls.call_args[1]
             assert call_kwargs.get("alpha") == 0.5
             assert call_kwargs.get("threshold") == 0.99
             print("Pysindy_Solver passed.")

def test_optimizer_integration():
    print("Testing Optimizer integration...")
    data = MagicMock(spec=DataConstructor)
    solver_path = "ddmo.solver.pysindy_solver.Pysindy_Solver" # not used directly here
    
    # Mock the solver function
    solver_func = MagicMock()
    solver_func.return_value = MagicMock(success=True)
    
    problem = Optimizer(
        data_instance=data,
        solver=solver_func,
        solver_param_names=["threshold"]
    )
    
    # Mock mapi_label which is usually set in main.py
    problem.mapi_label = {"a": 0, "b": 1, "threshold": 2}
    
    # Mock formulation 
    problem.formulation = None # Simplify to generic case first
    
    # Test _compute_metrics
    # vars: a=1, b=2, threshold=0.5
    vars = [1.0, 2.0, 0.5]
    
    # _compute_metrics calls self.solver(*solver_args, **solver_kwargs)
    result = problem._compute_metrics(vars)
    
    # Check if threshold was extracted and passed as kwarg
    solver_func.assert_called()
    call_args, call_kwargs = solver_func.call_args
    assert call_kwargs.get("threshold") == 0.5
    print("Optimizer integration passed.")

if __name__ == "__main__":
    try:
        test_optimizer_settings()
        test_pysindy_solver_kwargs()
        test_optimizer_integration()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
