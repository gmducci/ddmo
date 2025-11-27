# ddmo/formulation/weak_test_registry.py
from typing import Dict, Type
from ddmo.test_function_factory.base import TestFunctionFamily
from ddmo.test_function_factory.hann_window import HannWindow
# from ddmo.test_function_factory.gaussian import GaussianWindow, etc.

WEAK_TEST_FAMILY_REGISTRY: Dict[str, Type[TestFunctionFamily]] = {
    "HannWindow": HannWindow,
    # "GaussianWindow": GaussianWindow,
}