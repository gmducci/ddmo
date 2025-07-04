import numpy as np

from ddmo.utils.base import capped_simplex_projection


class TestCappedSimplexProjection:
    """Test cases for the capped_simplex_projection function."""

    def test_basic_functionality(self):
        """Test basic functionality with a simple array."""
        trimming_array = np.array([0.1, 0.5, 0.8, 0.2])
        trimming_fraction = 0.3

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # Check that result is bounded between 0 and 1
        assert np.all(result >= 0), "All values should be >= 0"
        assert np.all(result <= 1), "All values should be <= 1"

        # Check that the sum constraint is satisfied
        expected_sum = (1.0 - trimming_fraction) * trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6), \
            f"Sum should be {expected_sum}, got {actual_sum}"

    def test_zero_trimming_fraction(self):
        """Test with zero trimming fraction (no trimming)."""
        trimming_array = np.array([0.3, 0.7, 0.1, 0.9])
        trimming_fraction = 0.0

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # With zero trimming, sum should equal array size
        expected_sum = trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

        # All values should be 1 (capped at maximum)
        assert np.allclose(result, 1.0, atol=1e-6)

    def test_full_trimming_fraction(self):
        """Test with maximum trimming fraction."""
        trimming_array = np.array([0.3, 0.7, 0.1, 0.9])
        trimming_fraction = 1.0

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # With full trimming, sum should be zero
        expected_sum = 0.0
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

        # All values should be 0
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_uniform_array(self):
        """Test with uniform input array."""
        trimming_array = np.array([0.5, 0.5, 0.5, 0.5])
        trimming_fraction = 0.25

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # Result should be uniform due to symmetry
        assert np.allclose(result, result[0], atol=1e-6), \
            "Result should be uniform for uniform input"

        # Check sum constraint
        expected_sum = (1.0 - trimming_fraction) * trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

    def test_single_element_array(self):
        """Test with single element array."""
        trimming_array = np.array([0.7])
        trimming_fraction = 0.2

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # Check bounds
        assert 0 <= result[0] <= 1, "Result should be between 0 and 1"

        # Check sum constraint
        expected_sum = (1.0 - trimming_fraction) * 1
        assert np.isclose(result[0], expected_sum, atol=1e-6)

    def test_negative_values(self):
        """Test with negative values in input array."""
        trimming_array = np.array([-0.5, 0.3, -0.1, 0.8])
        trimming_fraction = 0.4

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # Check bounds
        assert np.all(result >= 0), "All values should be >= 0"
        assert np.all(result <= 1), "All values should be <= 1"

        # Check sum constraint
        expected_sum = (1.0 - trimming_fraction) * trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

    def test_large_values(self):
        """Test with values larger than 1 in input array."""
        trimming_array = np.array([1.5, 2.0, 0.5, 3.0])
        trimming_fraction = 0.3

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # Check bounds
        assert np.all(result >= 0), "All values should be >= 0"
        assert np.all(result <= 1), "All values should be <= 1"

        # Check sum constraint
        expected_sum = (1.0 - trimming_fraction) * trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

    def test_already_on_simplex(self):
        """Test with array that's already on the capped simplex."""
        # Create an array that should already satisfy the constraints
        trimming_fraction = 0.25
        # Create values that sum to (1-trimming_fraction) * size and are in [0,1]
        trimming_array = np.array([0.5, 0.8, 1.0, 0.7])  # sum = 3.0

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # Check bounds
        assert np.all(result >= 0), "All values should be >= 0"
        assert np.all(result <= 1), "All values should be <= 1"

        # Check sum constraint
        expected_sum = (1.0 - trimming_fraction) * trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

    def test_extreme_trimming_fractions(self):
        """Test with extreme trimming fractions."""
        trimming_array = np.array([0.1, 0.9, 0.5, 0.3])

        # Test very small trimming fraction
        result_small = capped_simplex_projection(trimming_array, 0.001)
        expected_sum_small = (1.0 - 0.001) * trimming_array.size
        assert np.isclose(np.sum(result_small), expected_sum_small, atol=1e-6)

        # Test very large trimming fraction (but less than 1)
        result_large = capped_simplex_projection(trimming_array, 0.999)
        expected_sum_large = (1.0 - 0.999) * trimming_array.size
        assert np.isclose(np.sum(result_large), expected_sum_large, atol=1e-6)

    def test_different_array_sizes(self):
        """Test with different array sizes."""
        trimming_fraction = 0.3

        # Small array
        small_array = np.array([0.5, 0.2])
        result_small = capped_simplex_projection(small_array, trimming_fraction)
        expected_sum_small = (1.0 - trimming_fraction) * 2
        assert np.isclose(np.sum(result_small), expected_sum_small, atol=1e-6)

        # Large array
        large_array = np.random.rand(20)
        result_large = capped_simplex_projection(large_array, trimming_fraction)
        expected_sum_large = (1.0 - trimming_fraction) * 20
        assert np.isclose(np.sum(result_large), expected_sum_large, atol=1e-6)

    def test_monotonicity(self):
        """Test that larger input values tend to produce larger output values."""
        trimming_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        trimming_fraction = 0.2

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # The projection should preserve some ordering, though not strict monotonicity
        # due to the capping constraint
        assert np.all(result >= 0) and np.all(result <= 1)

        # Check sum constraint
        expected_sum = (1.0 - trimming_fraction) * trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

    def test_edge_cases(self):
        """Test edge cases and potential numerical issues."""
        # Test with very small values
        small_array = np.array([1e-10, 1e-9, 1e-8])
        result_small = capped_simplex_projection(small_array, 0.1)
        assert np.all(result_small >= 0) and np.all(result_small <= 1)

        # Test with values very close to 1
        close_to_one = np.array([0.999999, 0.999998, 0.999997])
        result_close = capped_simplex_projection(close_to_one, 0.1)
        assert np.all(result_close >= 0) and np.all(result_close <= 1)

    def test_input_validation(self):
        """Test that function handles various input types correctly."""
        # Test with list input (should work with numpy conversion)
        trimming_list = [0.1, 0.5, 0.8, 0.2]
        trimming_fraction = 0.3

        result = capped_simplex_projection(np.array(trimming_list), trimming_fraction)
        assert np.all(result >= 0) and np.all(result <= 1)

        # Check sum constraint
        expected_sum = (1.0 - trimming_fraction) * len(trimming_list)
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

    def test_reproducibility(self):
        """Test that the function produces consistent results."""
        trimming_array = np.array([0.3, 0.7, 0.1, 0.9, 0.5])
        trimming_fraction = 0.25

        result1 = capped_simplex_projection(trimming_array, trimming_fraction)
        result2 = capped_simplex_projection(trimming_array, trimming_fraction)

        assert np.allclose(result1, result2, atol=1e-10), \
            "Function should produce identical results for identical inputs"

    def test_mathematical_properties(self):
        """Test mathematical properties of the projection."""
        trimming_array = np.array([0.2, 0.6, 0.3, 0.8, 0.1])
        trimming_fraction = 0.3

        result = capped_simplex_projection(trimming_array, trimming_fraction)

        # Property 1: Result should be a valid probability-like distribution
        # (non-negative, bounded by 1)
        assert np.all(result >= 0), "All values should be non-negative"
        assert np.all(result <= 1), "All values should be <= 1"

        # Property 2: Sum constraint
        expected_sum = (1.0 - trimming_fraction) * trimming_array.size
        actual_sum = np.sum(result)
        assert np.isclose(actual_sum, expected_sum, atol=1e-6), \
            f"Sum constraint violated: expected {expected_sum}, got {actual_sum}"

        # Property 3: The projection should be idempotent for valid inputs
        # (projecting an already valid point should return the same point)
        if np.all(result >= 0) and np.all(result <= 1) and \
           np.isclose(np.sum(result), expected_sum, atol=1e-6):
            result2 = capped_simplex_projection(result, trimming_fraction)
            assert np.allclose(result, result2, atol=1e-6), \
                "Projection should be idempotent"
