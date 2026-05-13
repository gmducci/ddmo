import numpy as np
import pytest

from ddmo.formulation.formulation_settings import FormulationSettings
from ddmo.library_functions import LibraryBuilder
from ddmo.solver.pysindy_solver import Pysindy_Solver
from ddmo.utils.data_constructor import DataConstructor


class TestDDMO:
    @pytest.fixture
    def synthetic_data(self):
        # Create a simple decaying exponential system: dx/dt = -0.5 * x
        t = np.linspace(0, 10, 100)
        x = np.exp(-0.5 * t).reshape(-1, 1)
        x_dot = -0.5 * x

        # Build a simple library
        library = LibraryBuilder().add_polynomial(max_degree=1)

        return DataConstructor(
            x_train=x,
            time_array=t,
            x_dot=x_dot,
            library_object=library.build,
            x_0=[x[0]],
            x_raw=x,
        )

    def test_smoke_identification(self, synthetic_data):
        """
        Smoke test to ensure the whole pipeline runs without crashing on simple data.
        """
        from ddmo.main import identify
        from ddmo.solver.settings import SolverSettings
        from ddmo.utils.constructors import OptimizerSettings

        # Formulation parameters (for weak formulation)
        formulation_bounds = {
            "window_length": (5, 20),
            "stride": (1.0, 2.0),
        }

        optimizer_settings = OptimizerSettings(
            formulation_bounds=formulation_bounds,
            library_bounds={},
            n_obj=2,
            n_gen=2,  # Minimal generations for speed
            pop_size=5,  # Minimal population for speed
            algorithm="NSGA2",
        )

        solver_settings = SolverSettings(
            solver_name="pysindy",
            regularization="STLSQ",
            regularization_parameters={"threshold": 0.01, "alpha": 0.05},
        )

        formulation_settings = FormulationSettings(
            formulation="weak",
            test_function="PolynomialWindow",
            degree=2,
            window_length=11,  # Initial value (must be odd usually)
            stride=1,
        )

        # Run identification
        res = identify(
            data_instance=synthetic_data,
            optimizer_settings=optimizer_settings,
            solver_settings=solver_settings,
            formulation_settings=formulation_settings,
        )

        assert res is not None
        assert res.objective_space is not None

    def test_solver_timeout_integration(self, synthetic_data):
        """
        Integration test: verify Pysindy_Solver.solve returns failed SolverResult on timeout.
        """
        from unittest.mock import patch

        # Mock simulate to raise TimeoutError instantly
        with patch(
            'ddmo.solver.pysindy_solver._ManualSINDyModel.simulate',
            side_effect=TimeoutError("Mock Timeout"),
        ):
            solver = Pysindy_Solver(
                data_instance=synthetic_data,
                optimizer="STLSQ",
                solver_parameters={"threshold": 0.01, "alpha": 0.05},
            )

            # Construct dummy payloads
            lhs = np.random.rand(10, 1)
            # Need a valid rhs dict with theta
            theta = np.random.rand(10, 1)
            rhs = {
                "opt_vars": [10, 1],
                "theta": theta,
                "library": None,
                "named_funcs": None,
                "colmask": None,
            }

            res = solver.solve(lhs, rhs)

            # Should return failed SolverResult
            assert not res.success
            assert res.solution is None
            assert res.coefficients is None
