import pytest

from ddmo.utils.constructors import IdentificationResult, OptimizerSettings


class TestOptimizerSettings:
    def test_objective_count_is_optional(self):
        settings = OptimizerSettings(
            library_bounds={},
            formulation_bounds={},
            n_gen=5,
        )
        assert settings.n_obj is None

    def test_initialization_valid(self):
        """Test valid initialization of OptimizerSettings."""
        settings = OptimizerSettings(
            library_bounds={"gamma": (0.1, 1.0)},
            formulation_bounds={"window": (5, 10)},
            n_obj=2,
            n_gen=50,
        )
        assert settings.n_obj == 2
        assert settings.n_gen == 50
        assert settings.algorithm == "NSGA2"  # Default
        assert settings.pop_size == 100  # Default

    def test_bounds_properties(self):
        """Test property accessors for bounds."""
        settings = OptimizerSettings(
            library_bounds={"a": (1, 2)},
            formulation_bounds={"b": (3, 4)},
            n_obj=1,
            n_gen=10,
        )

        # Test all_bounds
        assert settings.all_bounds == {"a": (1, 2), "b": (3, 4)}

        # Test xl (lower bounds)
        assert settings.xl == {"a": 1, "b": 3}

        # Test xu (upper bounds)
        assert settings.xu == {"a": 2, "b": 4}

        # Test n_var
        assert settings.n_var == 2

        # Test parameter names
        assert "a" in settings.library_param_names
        assert "b" in settings.formulation_param_names

    def test_invalid_bounds_format_not_tuple(self):
        """Test that non-tuple bounds raise ValueError."""
        with pytest.raises(ValueError, match="must be a .* tuple"):
            OptimizerSettings(
                library_bounds={"a": [1, 2]},  # List instead of tuple
                formulation_bounds={},
                n_obj=1,
                n_gen=10,
            )

    def test_invalid_bounds_tuple_length(self):
        """Test that tuples of wrong length raise ValueError."""
        with pytest.raises(ValueError, match="must be a .* tuple"):
            OptimizerSettings(
                library_bounds={"a": (1, 2, 3)},
                formulation_bounds={},
                n_obj=1,
                n_gen=10,
            )

    def test_invalid_bounds_min_greater_max(self):
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError, match="exceeds upper bound"):
            OptimizerSettings(
                library_bounds={"a": (2, 1)}, formulation_bounds={}, n_obj=1, n_gen=10
            )


class TestIdentificationResult:
    def test_instantiation(self):
        """Test simple instantiation of IdentificationResult."""
        res = IdentificationResult(
            problem="prob",
            result="res",
            objective_space=[],
            solutions=[],
            coeff_matrices=[],
            alphas=[],
            models=[],
            theta_matrices=[],
        )
        assert res.problem == "prob"
        assert res.solutions == []
