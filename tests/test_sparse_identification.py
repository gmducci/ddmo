import numpy as np
import pytest
from unittest.mock import patch
from ddmo.optimizer.optimizer import SparseIdentification

# Dummy library to mock the symbolic functions
class DummyLibrary:
    def _functions(self, alpha):
        return None, ["x", "y"], ["x + y"]

# Fixture to prepare test inputs
@pytest.fixture
def dummy_inputs():
    x_train = np.random.rand(10, 2)
    time_array = np.linspace(0, 1, 10)
    x_dot = np.gradient(x_train, axis=0)
    x_0 = [x_train[0]]
    x_raw = x_train.copy()
    lib = DummyLibrary()
    return {
        "x_train": x_train,
        "time_array": time_array,
        "x_dot": x_dot,
        "x_0": x_0,
        "x_raw": x_raw,
        "library_object": lib
    }

# Test initialization
def test_initialization(dummy_inputs):
    obj = SparseIdentification(**dummy_inputs, regression_model="lasso", alpha_lasso=0.01)
    assert obj.dimension == 2
    assert obj.number_initial_conditions == 1
    assert obj.target == "trajectory"

# Test regression model selection
def test_regression_model_selection(dummy_inputs):
    obj = SparseIdentification(**dummy_inputs, regression_model="ridge", alpha_lasso=0.1)
    model = obj._select_regression_model()
    assert model.__class__.__name__ == "Ridge"

# Test evaluation using mocks
@patch("ddmo.optimizer.optimizer.compute_trajectory")
@patch("ddmo.optimizer.optimizer.compute_mse", return_value=1.23)
def test_evaluate(mock_mse, mock_trajectory, dummy_inputs):
    dummy_solution = np.random.rand(10, 2)
    dummy_coeff = np.random.rand(2, 2)
    dummy_alpha = np.random.rand(2)
    mock_trajectory.return_value = (dummy_solution, dummy_solution, dummy_coeff, dummy_alpha)

    obj = SparseIdentification(**dummy_inputs, regression_model="lasso", alpha_lasso=0.01)
    x = np.random.rand(4, 2)

    out = {}
    obj._evaluate(x, out)

    assert "F" in out
    assert "solution" in out
    assert "coeff_matrix" in out
    assert "alphas" in out
    assert "dw_coeff" in out
    assert out["F"].shape == (4, 2)