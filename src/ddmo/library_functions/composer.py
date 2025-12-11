"""
LibraryBuilder: Fluent API for composing function libraries from factories.

Example:
    library = (
        LibraryBuilder()
        .add_polynomial(max_degree=3)
        .add_hill(exponent_param="gamma_1")
        .add_exponential(rate_param="gamma_2")
        .add_custom(lambda x, p: x**p["gamma_3"], "x^γ₃")
    )
    
    # Use in DataConstructor
    training_data = DataConstructor(
        library_object=library.build,
        ...
    )
"""
from dataclasses import dataclass, field
from typing import Callable, Any

from ddmo.library_functions.factories import (
    LibraryFactory,
    PolynomialFactory,
    HillFactory,
    ExponentialFactory,
    IdentityFactory,
    NamedFunc,
)


# Type for custom functions: takes (x, params_dict) -> float
CustomFunc = Callable[[float, dict[str, float]], float]
CustomNameFunc = Callable[[str, dict[str, float]], str] | str


@dataclass
class LibraryBuilder:
    """
    Fluent API for building function libraries from composable factories.
    
    Combines reusable factories (polynomial, Hill, exponential) with
    custom parameterized functions into a single library.
    
    The `build` method returns a callable that takes optimizer params
    and returns a list of (function, name) tuples compatible with PySINDy.
    
    Example:
        # Define library structure
        library = (
            LibraryBuilder()
            .add_polynomial(max_degree=2)
            .add_hill(exponent_param="gamma_1")
        )
        
        # Use in DataConstructor
        training_data = DataConstructor(
            library_object=library.build,
            ...
        )
        
        # Called by optimizer with current params:
        # library.build({"gamma_1": 9.0}) -> [(func, name), ...]
    """
    _factories: list[LibraryFactory] = field(default_factory=list)
    _custom_funcs: list[tuple[CustomFunc, CustomNameFunc]] = field(default_factory=list)
    
    def add_polynomial(self, max_degree: int = 2, include_constant: bool = False) -> "LibraryBuilder":
        """
        Add polynomial functions: x, x², x³, ...
        
        Args:
            max_degree: Maximum polynomial degree (e.g., 3 for x, x², x³).
            include_constant: Whether to include constant term (1).
            
        Returns:
            Self for method chaining.
        """
        self._factories.append(PolynomialFactory(max_degree=max_degree, include_constant=include_constant))
        return self
    
    def add_identity(self) -> "LibraryBuilder":
        """
        Add identity function: x
        
        Convenience method when you only want x without higher powers.
        
        Returns:
            Self for method chaining.
        """
        self._factories.append(IdentityFactory())
        return self
    
    def add_hill(self, exponent_param: str) -> "LibraryBuilder":
        """
        Add Hill function: 1 / (1 + x^γ)
        
        The exponent γ is read from optimizer parameters at runtime.
        
        Args:
            exponent_param: Name of the optimizer parameter for the exponent.
            
        Returns:
            Self for method chaining.
        """
        self._factories.append(HillFactory(exponent_param=exponent_param))
        return self
    
    def add_exponential(self, rate_param: str) -> "LibraryBuilder":
        """
        Add exponential decay function: exp(-k * x)
        
        The rate k is read from optimizer parameters at runtime.
        
        Args:
            rate_param: Name of the optimizer parameter for the rate.
            
        Returns:
            Self for method chaining.
        """
        self._factories.append(ExponentialFactory(rate_param=rate_param))
        return self
    
    def add_factory(self, factory: LibraryFactory) -> "LibraryBuilder":
        """
        Add a custom factory instance.
        
        Args:
            factory: Any LibraryFactory subclass instance.
            
        Returns:
            Self for method chaining.
        """
        self._factories.append(factory)
        return self
    
    def add_custom(
        self, 
        func: CustomFunc, 
        name: CustomNameFunc
    ) -> "LibraryBuilder":
        """
        Add a custom parameterized function.
        
        Args:
            func: Function taking (x, params_dict) -> float.
            name: Either a string or callable taking (varname, params_dict) -> str.
            
        Returns:
            Self for method chaining.
            
        Example:
            .add_custom(
                lambda x, p: x ** p["gamma_3"],
                lambda x, p: f"{x}^{p['gamma_3']:.2f}"
            )
        """
        self._custom_funcs.append((func, name))
        return self
    
    def build(self, params: dict[str, float]) -> list[NamedFunc]:
        """
        Build the function library with given optimizer parameters.
        
        This method is called by the formulation on each optimizer evaluation.
        
        Args:
            params: Dictionary of current optimizer parameter values.
            
        Returns:
            List of (function, name_generator) tuples for PySINDy.
        """
        result: list[NamedFunc] = []
        
        # Generate from all factories
        for factory in self._factories:
            result.extend(factory.generate(params))
        
        # Add custom functions
        for func, name in self._custom_funcs:
            # Wrap custom function to match NamedFunc signature
            wrapped_func = self._wrap_custom_func(func, params)
            wrapped_name = self._wrap_custom_name(name, params)
            result.append((wrapped_func, wrapped_name))
        
        return result
    
    @staticmethod
    def _wrap_custom_func(func: CustomFunc, params: dict[str, float]) -> Callable[[float], float]:
        """Wrap a custom function to capture params."""
        return lambda x, f=func, p=params: f(x, p)
    
    @staticmethod
    def _wrap_custom_name(name: CustomNameFunc, params: dict[str, float]) -> Callable[[str], str]:
        """Wrap a custom name to capture params."""
        if isinstance(name, str):
            return lambda x, n=name: n
        return lambda x, n=name, p=params: n(x, p)
