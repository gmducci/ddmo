# ddmo/test_function_factory/gaussian_window.py

from __future__ import annotations

import numpy as np

from .base import Array, TestFunctionFamily


class GaussianWindow(TestFunctionFamily):
    """
    Gaussian-window test-function family on a 1D time grid.

    Parameters (in params)
    ----------------------
    window_length : int
        Length of each window (odd integer ≥ 3), in grid points.
    stride : int
        Step between consecutive window centres, in grid points.
    sigma : float
        Standard deviation of the Gaussian, expressed in grid points.
        (i.e. in the index variable, not physical time).
    """

    name = "GaussianWindow"
    allowed_params = {"window_length", "stride", "sigma"}

    def _validate_params(self) -> None:
        # read parameters from the base-class params dict
        try:
            window_length = int(self.params["window_length"])
            stride = int(self.params["stride"])
            sigma = float(self.params["sigma"])
        except KeyError as e:
            raise ValueError(
                "GaussianWindow requires parameters "
                "'window_length', 'stride', and 'sigma' in params."
            ) from e

        if window_length % 2 == 0 or window_length < 3:
            raise ValueError(
                f"window_length must be an odd integer ≥ 3, got {window_length}."
            )
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}.")

        self.window_length = window_length
        self.stride = stride
        self.sigma = sigma
        self.half = window_length // 2

        # build prototype Gaussian window (unscaled)
        w = self._gaussian(self.window_length, self.sigma)

        # numerical derivative dw/dt (unscaled)
        # here dt is the physical time step; sigma is in grid indices
        dt = float(self.dt)
        dw = self.finite_difference(w, dt=dt, axis=-1)

        # normalisation so that ∑ phi * dt ≈ 1
        base_alpha = 1.0 / (np.sum(w) * dt)
        total_alpha = self.alpha * base_alpha

        self.phi = w * total_alpha
        self.dphi = dw * total_alpha

        self._centers = None

    @staticmethod
    def _gaussian(L: int, sigma: float) -> Array:
        """
        Unnormalised Gaussian on integer grid indices centred at 0.
        """
        if L % 2 == 0 or L < 3:
            raise ValueError("Gaussian length must be an odd integer ≥ 3.")
        k = np.arange(L) - (L // 2)
        return np.exp(-0.5 * (k / sigma) ** 2)

    # same grid_centers/build_tests as HannWindow, specialised to 1D

    @property
    def centers(self) -> Array:
        if self._centers is None:
            raise RuntimeError(
                "centers have not been computed yet; "
                "call grid_centers(M) or build_tests(M) first."
            )
        return self._centers

    def grid_centers(self, M: int) -> Array:
        """
        Compute centres for a 1D grid of length M, ensuring full support
        of each window inside [0, M-1].
        """
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
                f"GaussianWindow.build_tests only supports 1D grids, got {grid_shape}."
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
