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

from collections.abc import Callable
from dataclasses import dataclass, field

from ddmo.library_functions.factories import (
    CrossProductFactory,
    CustomFunctionFactory,
    ExponentialFactory,
    HillFactory,
    IdentityFactory,
    LibraryFactory,
    NamedFunc,
    PolynomialFactory,
    PowerCrossProductFactory,
)

# Type for custom functions: takes (*args, params_dict) -> float
CustomFunc = Callable[..., float]
CustomNameFunc = Callable[..., str] | str


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
    _static_funcs: list[tuple[Callable, Callable]] = field(default_factory=list)

    def add_polynomial(
        self, max_degree: int = 2, include_constant: bool = False, include_interactions: bool = True
    ) -> "LibraryBuilder":
        """
        Add polynomial functions: x, x², x³, ...

        Args:
            max_degree: Maximum polynomial degree (e.g., 3 for x, x², x³).
            include_constant: Whether to include constant term (1).
            include_interactions: Whether to include cross-product terms (x*y).

        Returns:
            Self for method chaining.
        """
        self._factories.append(
            PolynomialFactory(max_degree=max_degree, include_constant=include_constant, include_interactions=include_interactions)
        )
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

    def add_cross_product(self) -> "LibraryBuilder":
        """
        Add cross-product interaction term: x * y

        PySINDy will apply this to all pairs of state variables,
        generating terms like x₀*x₁, x₀*x₂, x₁*x₂, etc.

        Returns:
            Self for method chaining.
        """
        self._factories.append(CrossProductFactory())
        return self

    def add_power_cross_product(
        self, power_x: int = 1, power_y: int = 1
    ) -> "LibraryBuilder":
        """
        Add powered cross-product: x^a * y^b

        Args:
            power_x: Power for first variable.
            power_y: Power for second variable.

        Returns:
            Self for method chaining.

        Example:
            .add_power_cross_product(2, 1)  # Generates x²*y
        """
        self._factories.append(
            PowerCrossProductFactory(power_x=power_x, power_y=power_y)
        )
        return self

    def add_parametric_power_cross_product(
        self, exponent_x_param: str, exponent_y_param: str
    ) -> "LibraryBuilder":
        """
        Add parametric powered cross-product: x^γ₁ * y^γ₂

        Args:
            exponent_x_param: Name of the optimizer parameter for γ₁.
            exponent_y_param: Name of the optimizer parameter for γ₂.

        Returns:
            Self for method chaining.
        """

        def func(x, y, p):
            return (x ** p[exponent_x_param]) * (y ** p[exponent_y_param])

        def name(x, y, p):
            gx = p[exponent_x_param]
            gy = p[exponent_y_param]
            return f"{x}^{gx:.2f}*{y}^{gy:.2f}"

        return self.add_custom(func, name)

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

    def add_function(
        self, func: Callable[[float], float], name: str | Callable[[str], str]
    ) -> "LibraryBuilder":
        """
        Add a simple static function (doesn't depend on optimizer params).

        Args:
            func: Function taking x -> float (e.g., lambda x: np.sin(x)).
            name: String name or callable taking varname -> str.

        Returns:
            Self for method chaining.

        Example:
            .add_function(lambda x: np.sin(x), "sin")
            .add_function(lambda x: np.cos(x), lambda x: f"cos({x})")
        """
        # Wrap as a NamedFunc and store directly
        if isinstance(name, str):
            name_func = lambda x, n=name: f"{n}({x})"
        else:
            name_func = name
        self._static_funcs.append((func, name_func))
        return self

    def add_custom(self, func: CustomFunc, name: CustomNameFunc) -> "LibraryBuilder":
        """
        Add a custom parameterized function (depends on optimizer params).

        Args:
            func: Function taking (*args, params_dict) -> float.
            name: Either a string or callable taking (*args, params_dict) -> str.

        Returns:
            Self for method chaining.

        Example:
            .add_custom(
                lambda x, p: x ** p["gamma_3"],
                lambda x, p: f"{x}^{p['gamma_3']:.2f}"
            )
        """
        self._factories.append(CustomFunctionFactory(func=func, name=name))
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

        # Add static functions (don't depend on params)
        for func, name_func in self._static_funcs:
            result.append((func, name_func))

        return result
