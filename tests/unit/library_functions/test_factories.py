import pytest
import numpy as np
from ddmo.library_functions.factories import (
    PolynomialFactory,
    HillFactory,
    ExponentialFactory,
    CrossProductFactory,
    PowerCrossProductFactory,
    IdentityFactory
)

class TestPolynomialFactory:
    def test_polynomial_defaults(self):
        """Test default polynomial generation (max_degree=2)."""
        factory = PolynomialFactory()
        funcs = factory.generate({})
        
        # Expected: x, x^2 (interactions disabled by logic if input dim unknown? 
        # Actually generate produces logic, but let's check what it produces.
        # Looking at code:
        # loops degree 1..max_degree
        # loops loops interactions...
        # Wait, the factory logic generates functions that *take* arguments.
        # But generate() doesn't know input dimension, it just returns functions.
        # Ah, looking at the code: 
        # It generates single variable keys? No, PySINDy applies them.
        # Let's check the generate method again.
        # It yields (func, name).
        
        # For degree=1: x
        # For degree=2: x^2
        # Interactions: (px=1, py=1) -> x*y
        
        # Total expected: 3 functions.
        assert len(funcs) == 3
        
        # Check execution
        f1, n1 = funcs[0]
        assert f1(2) == 2
        assert n1("x") == "x"
        
        f2, n2 = funcs[1] # x^2
        assert f2(2) == 4
        assert n2("x") == "x^2"
        
        f3, n3 = funcs[2] # x*y
        assert f3(2, 3) == 6
        assert n3("x", "y") == "x*y"

    def test_polynomial_degree_3(self):
        factory = PolynomialFactory(max_degree=3)
        funcs = factory.generate({})
        # degree 1: x
        # degree 2: x^2
        # degree 3: x^3
        # interactions deg 2: x*y
        # interactions deg 3: x^2*y, x*y^2
        assert len(funcs) == 6

    def test_constant_term(self):
        factory = PolynomialFactory(max_degree=1, include_constant=True, include_interactions=False)
        funcs = factory.generate({})
        assert len(funcs) == 2 # 1, x
        assert funcs[0][0](100) == 1.0
        assert funcs[0][1]("x") == "1"


class TestHillFactory:
    def test_hill_generation(self):
        # 1 / (1 + x^gamma)
        factory = HillFactory(exponent_param="g")
        funcs = factory.generate({"g": 2.0})
        
        assert len(funcs) == 1
        f, n = funcs[0]
        
        # 1 / (1 + 2^2) = 1/5 = 0.2
        assert f(2.0) == 0.2
        assert n("u") == "1/(1+u^2.00)"

    def test_missing_param_error(self):
        factory = HillFactory(exponent_param="g")
        with pytest.raises(ValueError, match="Missing parameter"):
            factory.generate({})


class TestExponentialFactory:
    def test_exponential_generation(self):
        # exp(-k * x)
        factory = ExponentialFactory(rate_param="k")
        funcs = factory.generate({"k": 0.5})
        
        assert len(funcs) == 1
        f, n = funcs[0]
        
        # exp(-0.5 * 0) = 1
        assert f(0.0) == 1.0
        # exp(-0.5 * 2) = exp(-1) = 0.3678...
        assert np.isclose(f(2.0), np.exp(-1))
        assert n("u") == "exp(-0.50*u)"


class TestCrossProductFactory:
    def test_cross_product(self):
        factory = CrossProductFactory()
        funcs = factory.generate({})
        assert len(funcs) == 1
        f, n = funcs[0]
        assert f(2, 3) == 6
        assert n("x", "y") == "x*y"


class TestPowerCrossProductFactory:
    def test_power_cross_product(self):
        factory = PowerCrossProductFactory(power_x=2, power_y=1) # x^2 * y
        funcs = factory.generate({})
        assert len(funcs) == 1
        f, n = funcs[0]
        assert f(3, 2) == 9 * 2 # 18
        assert n("a", "b") == "a^2*b"
