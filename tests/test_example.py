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

    def test_solver_timeout(self, synthetic_data):
        """
        Test that the solver correctly handles timeouts during simulation.
        We force a timeout by using a very short timeout value.
        """
        import pysindy as ps

        from ddmo.solver.pysindy_solver import _ManualSINDyModel

        # Dummy library and coefficients
        lib = ps.CustomLibrary(library_functions=[lambda x: x])
        lib.fit(np.array([[1.0]]))  # Fit to establish n_features_in_

        coefs = np.array([[-0.5]])  # dx/dt = -0.5x
        mask = np.array([True])

        model = _ManualSINDyModel(
            feature_library=lib,
            coefficients=coefs,
            optimizer=None,
            solver_kwargs={"method": "RK45"},
            active_mask=mask,
        )

        # Test immediate timeout
        # Using a timeout of 1 second but requesting a massive simulation
        with pytest.raises(TimeoutError):
            # We simulate a large time range with small steps relative to the range
            # to ensure it takes enough time.
            # timeout=0 disables alarm in signal.alarm(0), so we use 1 with huge task
            # if 1s is too long, we can try mocking signal.alarm if needed,
            # but let's try a real timeout first or mock the handler.

            # Actually, waiting 1s in unit test is slow.
            # Better to mock signal.alarm or simply rely on "simulate" implementation detail check.
            # But here we want to verifying the "simulate" method raises TimeoutError.

            # Let's mock time.sleep inside the rhs or make rhs extremely slow?
            # Or just allow 1s wait.

            # For speed, let's mock signal.signal/alarm behavior?
            # No, let's just make the function sleep inside the integration step?
            # solve_ivp calls RHS.

            model.simulate(
                x0=np.array([1.0]),
                time_array=np.linspace(0, 1e6, int(1e6)),  # Huge task
                timeout=1,
            )

    def test_solver_timeout_integration(self, synthetic_data):
        """
        Integration test: verify Pysindy_Solver.solve returns None on timeout.
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

            # Should return None tuple
            assert res == (None, None, None, None, None)
