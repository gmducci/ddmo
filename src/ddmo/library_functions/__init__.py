"""
Library Functions module for building SINDy-compatible function libraries.

Main API:
    from ddmo.library_functions import LibraryBuilder
    
    library = (
        LibraryBuilder()
        .add_polynomial(max_degree=3)
        .add_cross_product()               # x*y interaction terms
        .add_hill(exponent_param="gamma_1")
        .add_exponential(rate_param="gamma_2")
    )
"""
from ddmo.library_functions.factories import (
    LibraryFactory,
    PolynomialFactory,
    HillFactory,
    ExponentialFactory,
    IdentityFactory,
    CrossProductFactory,
    PowerCrossProductFactory,
    NamedFunc,
)
from ddmo.library_functions.composer import LibraryBuilder

__all__ = [
    # Main API
    "LibraryBuilder",
    # Factories
    "LibraryFactory",
    "PolynomialFactory",
    "HillFactory",
    "ExponentialFactory",
    "IdentityFactory",
    "CrossProductFactory",
    "PowerCrossProductFactory",
    # Types
    "NamedFunc",
]
