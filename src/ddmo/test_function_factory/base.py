# base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class TestFunctionBase(ABC):
    """
    Abstract template class for generic test functions (windows) and their derivatives.

    Subclasses only need to implement:
        • the construction of the base function φ via `evaluate`
        • any function-specific parameters (length, stride, etc.)

    This class provides:
        • storage for φ and its derivative
        • a generic finite-difference derivative
        • automatic scaling by α
    """

    name: str = "TestFunctionBase"

    def __init__(self, dt: float = 1.0, alpha: float = 1.0) -> None:
        self.dt = float(dt)
        self.alpha = float(alpha)

        self.test_function: Optional[np.ndarray] = None
        self.derivative_test_function: Optional[np.ndarray] = None

        self.phi: Optional[np.ndarray] = None
        self.dphi: Optional[np.ndarray] = None

    @abstractmethod
    def evaluate(self, length: int) -> np.ndarray:
        """
        Return the unscaled test function φ of size `length`.
        Subclasses must override this method.
        """
        raise NotImplementedError

    @staticmethod
    def evaluate_derivative(phi: np.ndarray, dt: float) -> np.ndarray:
        """
        Central finite-difference approximation of dφ/dt.

        dφ[i] ≈ (φ[i+1] − φ[i−1]) / (2 dt)
        Endpoints are set to zero.
        """
        dphi = np.empty_like(phi)

        if phi.size < 3:
            dphi.fill(0.0)
            return dphi

        dphi[1:-1] = (phi[2:] - phi[:-2]) / (2.0 * dt)
        dphi[0] = 0.0
        dphi[-1] = 0.0
        return dphi

    def build(self, length: int) -> None:
        """
        Build and store:
            • self.test_function  (unscaled φ)
            • self.derivative_test_function  (unscaled dφ/dt)
            • self.phi  = α φ
            • self.dphi = α dφ/dt
        """
        self.test_function = self.evaluate(length)
        self.derivative_test_function = self.evaluate_derivative(
            self.test_function, self.dt
        )

        self.phi = self.test_function * self.alpha
        self.dphi = self.derivative_test_function * self.alpha

    def __call__(self, length: int) -> np.ndarray:
        """
        Allow calling the object as a function: f(L) = φ(L) unscaled.
        """
        return self.evaluate(length)


def window_derivative(phi: np.ndarray, dt: float) -> np.ndarray:
    """
    Optional helper that exposes the derivative function with
    the signature previously used by HannWindow.
    """
    return TestFunctionBase.evaluate_derivative(phi, dt)