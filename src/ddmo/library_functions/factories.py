"""
Factory classes for generating SINDy-compatible function libraries.

Each factory generates functions as list[NamedFunc] where:
    NamedFunc = tuple[Callable[[float], float], Callable[[str], str]]
    # (function, name_generator)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

# Type alias for named functions (compatible with PySINDy CustomLibrary)
NamedFunc = tuple[Callable[..., float], Callable[..., str]]


class LibraryFactory(ABC):
    """Abstract base class for function library factories."""
    
    @abstractmethod
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """
        Generate functions based on optimizer parameters.
        
        Args:
            params: Dictionary of optimizer parameter values.
            
        Returns:
            List of (function, name_generator) tuples.
        """
        pass


@dataclass
class PolynomialFactory(LibraryFactory):
    """
    Factory for polynomial functions including cross-product terms.
    
    Generates terms like: x, y, x², y², x*y, x³, x²*y, x*y², y³, ...
    This matches PySINDy's PolynomialLibrary behavior.
    
    Example:
        factory = PolynomialFactory(max_degree=2)
        # Generates single-var: x, x²
        # Plus two-var interactions: x*y (applied by PySINDy to all pairs)
    
    Args:
        max_degree: Maximum total polynomial degree.
        include_interactions: Whether to include cross-product terms (default True).
        include_constant: Whether to include constant term 1 (default False).
    """
    max_degree: int = 2
    include_interactions: bool = True
    include_constant: bool = False
    
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """Generate polynomial functions up to max_degree with interactions."""
        funcs: list[NamedFunc] = []
        
        # Optionally include constant term
        if self.include_constant:
            funcs.append((
                lambda x: 1.0,
                lambda x: "1"
            ))
        
        # Generate single-variable terms: x, x², x³, ...
        for degree in range(1, self.max_degree + 1):
            func = self._make_power_func(degree)
            name = self._make_power_name(degree)
            funcs.append((func, name))
        
        # Generate cross-product / interaction terms
        if self.include_interactions:
            # x*y (degree 2)
            if self.max_degree >= 2:
                funcs.append((lambda x, y: x * y, lambda x, y: f"{x}*{y}"))
            
            # x²*y, x*y² (degree 3)
            if self.max_degree >= 3:
                funcs.append((
                    self._make_power_cross_func(2, 1),
                    self._make_power_cross_name(2, 1)
                ))
                funcs.append((
                    self._make_power_cross_func(1, 2),
                    self._make_power_cross_name(1, 2)
                ))
            
            # x³*y, x²*y², x*y³ (degree 4)
            if self.max_degree >= 4:
                funcs.append((
                    self._make_power_cross_func(3, 1),
                    self._make_power_cross_name(3, 1)
                ))
                funcs.append((
                    self._make_power_cross_func(2, 2),
                    self._make_power_cross_name(2, 2)
                ))
                funcs.append((
                    self._make_power_cross_func(1, 3),
                    self._make_power_cross_name(1, 3)
                ))
        
        return funcs
    
    @staticmethod
    def _make_power_func(degree: int) -> Callable[[float], float]:
        """Create a power function with captured degree."""
        d = degree
        if d == 1:
            return lambda x: x
        return lambda x: x ** d
    
    @staticmethod
    def _make_power_name(degree: int) -> Callable[[str], str]:
        """Create a name generator for power functions."""
        d = degree
        if d == 1:
            return lambda x: x
        return lambda x: f"{x}^{d}"
    
    @staticmethod
    def _make_power_cross_func(px: int, py: int) -> Callable[[float, float], float]:
        """Create a powered cross-product function."""
        a, b = px, py
        return lambda x, y: (x ** a) * (y ** b)
    
    @staticmethod
    def _make_power_cross_name(px: int, py: int) -> Callable[[str, str], str]:
        """Create a name generator for powered cross-product."""
        a, b = px, py
        x_part = "x" if a == 1 else f"x^{a}"
        y_part = "y" if b == 1 else f"y^{b}"
        # Replace placeholders at call time
        return lambda x, y: f"{x_part}*{y_part}".replace("x", x).replace("y", y)


@dataclass
class HillFactory(LibraryFactory):
    """
    Factory for Hill functions: 1 / (1 + x^γ)
    
    The exponent γ is read from optimizer parameters.
    
    Example:
        factory = HillFactory(exponent_param="gamma_1")
        funcs = factory.generate({"gamma_1": 9.0})  # Returns 1/(1+x^9)
    """
    exponent_param: str
    
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """Generate Hill function with parameterized exponent."""
        if self.exponent_param not in params:
            raise ValueError(f"Missing parameter '{self.exponent_param}' for HillFactory")
        
        gamma = params[self.exponent_param]
        
        # Create function with proper closure (no default args - PySINDy inspects arity)
        def make_hill(g):
            return lambda x: 1.0 / (1.0 + x ** g)
        
        def make_hill_name(g):
            return lambda x: f"1/(1+{x}^{g:.2f})"
        
        return [(make_hill(gamma), make_hill_name(gamma))]


@dataclass
class ExponentialFactory(LibraryFactory):
    """
    Factory for exponential decay functions: exp(-k * x)
    
    The rate k is read from optimizer parameters.
    
    Example:
        factory = ExponentialFactory(rate_param="gamma_2")
        funcs = factory.generate({"gamma_2": 5.0})  # Returns exp(-5*x)
    """
    rate_param: str
    
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """Generate exponential function with parameterized rate."""
        if self.rate_param not in params:
            raise ValueError(f"Missing parameter '{self.rate_param}' for ExponentialFactory")
        
        k = params[self.rate_param]
        
        # Create function with proper closure (no default args - PySINDy inspects arity)
        def make_exp(rate):
            return lambda x: np.exp(-rate * x)
        
        def make_exp_name(rate):
            return lambda x: f"exp(-{rate:.2f}*{x})"
        
        return [(make_exp(k), make_exp_name(k))]


@dataclass  
class IdentityFactory(LibraryFactory):
    """
    Factory for identity function: x
    
    Convenience factory when you just want x without higher powers.
    """
    
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """Generate identity function."""
        return [(lambda x: x, lambda x: x)]


@dataclass
class CrossProductFactory(LibraryFactory):
    """
    Factory for cross-product / interaction terms: x*y
    
    Generates a two-variable function that multiplies its inputs.
    PySINDy will apply this to all pairs of state variables.
    
    Example:
        factory = CrossProductFactory()
        funcs = factory.generate({})  # Returns x*y (applied to all variable pairs)
    """
    
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """Generate cross-product function."""
        return [(lambda x, y: x * y, lambda x, y: f"{x}*{y}")]


@dataclass
class PowerCrossProductFactory(LibraryFactory):
    """
    Factory for powered cross-products: x^a * y^b
    
    Generates interaction terms with specified powers.
    
    Example:
        factory = PowerCrossProductFactory(power_x=2, power_y=1)
        funcs = factory.generate({})  # Returns x²*y
    """
    power_x: int = 1
    power_y: int = 1
    
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """Generate powered cross-product function."""
        px, py = self.power_x, self.power_y
        
        def make_func(a, b):
            return lambda x, y: (x ** a) * (y ** b)
        
        def make_name(a, b):
            x_part = "x" if a == 1 else f"x^{a}"
            y_part = "y" if b == 1 else f"y^{b}"
            return lambda x, y: f"{x_part}*{y_part}".replace("x", x).replace("y", y)
        
        return [(make_func(px, py), make_name(px, py))]

