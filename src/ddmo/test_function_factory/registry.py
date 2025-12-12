from ddmo.test_function_factory.gaussian_window import GaussianWindow
from ddmo.test_function_factory.hann_window import HannWindow
from ddmo.test_function_factory.polynomial_window import PolynomialWindow

WEAK_TEST_FAMILY_REGISTRY = {
    "HannWindow": HannWindow,
    "GaussianWindow": GaussianWindow,
    "PolynomialWindow": PolynomialWindow,
}
