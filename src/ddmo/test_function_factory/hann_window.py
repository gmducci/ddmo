# ddmo/test_function_factory/hann_window.py

from __future__ import annotations

import numpy as np

from .base import Array, TestFunctionFamily


class HannWindow(TestFunctionFamily):
    """
    Hann-window test-function family on a 1D time grid.

    Parameters (in params)
    ----------------------
    window_length : int
    stride : int
    """

    name = "HannWindow"
    allowed_params = {"window_length", "stride"}

    def _validate_params(self) -> None:
        try:
            window_length = int(self.params["window_length"])
            stride = int(self.params["stride"])
        except KeyError as e:
            raise ValueError(
                "HannWindow requires parameters 'window_length' and 'stride' in params."
            ) from e

        if window_length % 2 == 0 or window_length < 3:
            raise ValueError(
                f"window_length must be an odd integer ≥ 3, got {window_length}."
            )

        self.window_length = window_length
        self.stride = stride
        self.half = self.window_length // 2

        w = self._hann(self.window_length)

        dt = float(self.dt)
        dw = self.finite_difference(w, dt=dt, axis=-1)

        base_alpha = 1.0 / (np.sum(w) * dt)
        total_alpha = self.alpha * base_alpha

        self.phi = w * total_alpha
        self.dphi = dw * total_alpha

        self._centers = None

    @staticmethod
    def _hann(L: int) -> Array:
        if L % 2 == 0 or L < 3:
            raise ValueError("Hann length must be an odd integer ≥ 3.")
        n = np.arange(L)
        w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (L - 1))
        w[0] = 0.0
        w[-1] = 0.0
        return w

    @property
    def centers(self) -> Array:
        if self._centers is None:
            raise RuntimeError(
                "centers have not been computed yet; "
                "call grid_centers(M) or build_tests(M) first."
            )
        return self._centers

    def grid_centers(self, M: int) -> Array:
        M = int(M)
        if M <= self.window_length:
            raise ValueError(
                f"Grid too small (M={M}) for window_length={self.window_length}."
            )
        last_center = M - self.half - 1
        centers = np.arange(self.half, last_center + 1, self.stride, dtype=int)
        self._centers = centers
        return centers

    def build_tests(self, grid_shape: tuple[int, ...]) -> list[Array]:
        if len(grid_shape) != 1:
            raise ValueError(
                f"HannWindow.build_tests only supports 1D grids, got {grid_shape}."
            )
        M = int(grid_shape[0])
        centers = self.grid_centers(M)

        tests: list[Array] = []
        for c in centers:
            t = np.zeros(M, dtype=float)
            sl = slice(c - self.half, c + self.half + 1)
            t[sl] = self.phi
            tests.append(t)
        return tests
