import pytest
from ddmo.formulation.formulation_settings import FormulationSettings

class TestFormulationSettings:
    def test_strong_formulation_valid(self):
        """Test valid strong formulation initialization."""
        fs = FormulationSettings(formulation="strong")
        assert fs.formulation == "strong"
        
    def test_strong_formulation_generic_normalization(self):
        """Test case insensitivity."""
        fs = FormulationSettings(formulation="STRONG")
        assert fs.formulation == "strong"

    def test_weak_formulation_valid(self):
        """Test valid weak formulation initialization."""
        fs = FormulationSettings(
            formulation="weak",
            test_function="Poly",
            degree=2,
            window_length=5,
            stride=1
        )
        assert fs.formulation == "weak"
        assert fs.test_function == "Poly"

    def test_weak_formulation_missing_test_function(self):
        """Test that weak formulation requires test_function."""
        with pytest.raises(ValueError, match="requires 'test_function'"):
            FormulationSettings(formulation="weak")

    def test_invalid_formulation_name(self):
        """Test unknown formulation name raises error."""
        with pytest.raises(ValueError, match="Unknown formulation"):
            FormulationSettings(formulation="invalid_name")

    def test_to_attributes_dict(self):
        """Test conversion to attributes dictionary."""
        fs = FormulationSettings(
            formulation="weak",
            test_function="Poly",
            degree=2
        )
        attrs = fs.to_attributes_dict()
        assert attrs["test_functions_class"] == "Poly"
        assert attrs["degree"] == 2
        assert "window_length" not in attrs
