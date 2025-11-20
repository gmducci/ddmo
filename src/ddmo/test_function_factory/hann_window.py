# hann_window.py

from __future__ import annotations

import numpy as np

from .base import TestFunctionBase, window_derivative


class HannWindow(TestFunctionBase):
    """
    Concrete Hann window implementation based on the generic TestFunctionBase.

    Parameters
    ----------
    window_length : int
        Length of the window (must be an odd integer ≥ 3).
    stride : int
        Stride associated with the window (kept as metadata).
    dt : float, optional
        Sampling step used for the numerical derivative, by default 1.0.
    alpha : float, optional
        Scaling factor applied to the window and its derivative, by default 1.0.
    """

    name = "HannWindow"

    def __init__(
        self,
        window_length: int,
        stride: int,
        dt: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(dt=dt, alpha=alpha)

        if window_length % 2 == 0 or window_length < 3:
            raise ValueError("Window length must be an odd integer ≥ 3.")

        self.window_length = window_length
        self.stride = stride
        self.half = window_length // 2

        # Unscaled test function φ
        self.test_function = self.evaluate(self.window_length)

        # Unscaled derivative dφ/dt
        self.derivative_test_function = window_derivative(
            self.test_function, self.dt
        )

        # Scaled versions
        self.phi = self.test_function * self.alpha
        self.dphi = self.derivative_test_function * self.alpha

    @staticmethod
    def evaluate(length: int) -> np.ndarray:
        """
        Evaluate the (unscaled) Hann window of a given length.

        The endpoints are forced to zero explicitly.
        """
        if length % 2 == 0 or length < 3:
            raise ValueError("Window length must be an odd integer ≥ 3.")

        n = np.arange(length)
        w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (length - 1))
        w[0] = 0.0
        w[-1] = 0.0
        return w

    def __call__(self, length: int | None = None) -> np.ndarray:
        """
        Callable interface.

        If `length` is None, returns the stored (unscaled) test function φ
        of size `window_length`. Otherwise, recomputes the Hann window
        of the requested length.
        """
        if length is None:
            if self.test_function is None:
                raise ValueError("The window has not been built yet.")
            return self.test_function
        return self.evaluate(length)