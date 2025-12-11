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
    Factory for polynomial functions: x, x², x³, ...
    
    This is a static factory - it does not use optimizer parameters.
    
    Example:
        factory = PolynomialFactory(max_degree=3)
        funcs = factory.generate({})  # Returns x, x², x³
    """
    max_degree: int = 2
    include_constant: bool = False
    
    def generate(self, params: dict[str, float]) -> list[NamedFunc]:
        """Generate polynomial functions up to max_degree."""
        funcs: list[NamedFunc] = []
        
        # Optionally include constant term
        if self.include_constant:
            funcs.append((
                lambda x: 1.0,
                lambda x: "1"
            ))
        
        # Generate x, x², x³, ...
        for degree in range(1, self.max_degree + 1):
            # Capture degree in closure properly
            func = self._make_power_func(degree)
            name = self._make_power_name(degree)
            funcs.append((func, name))
        
        return funcs
    
    @staticmethod
    def _make_power_func(degree: int) -> Callable[[float], float]:
        """Create a power function with captured degree."""
        # Use closure without default args (PySINDy inspects arity)
        d = degree  # Capture in closure
        if d == 1:
            return lambda x: x
        return lambda x: x ** d
    
    @staticmethod
    def _make_power_name(degree: int) -> Callable[[str], str]:
        """Create a name generator for power functions."""
        d = degree  # Capture in closure
        if d == 1:
            return lambda x: x
        return lambda x: f"{x}^{d}"


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
