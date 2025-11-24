# ddmo/test_function_factory/hann_window.py

from __future__ import annotations
import numpy as np

from .base import TestFunctionFamily, Array


class HannWindow(TestFunctionFamily):
    """
    Hann-window test-function family on a 1D time grid.

    Parameters
    ----------
    window_length : int
        Length of each window (odd integer ≥ 3), in grid points.
    stride : int
        Step between consecutive window centres, in grid points.
    dt : float, optional
        Time step of the data. Used both for derivative and normalisation.
    alpha : float, optional
        Optional extra scaling factor (default 1.0). The effective scaling will be
        alpha * (1 / (sum(window) * dt)) so that ∑ phi * dt ≈ 1, as in your
        original WeakFormulation.
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
            raise ValueError("window_length must be an odd integer ≥ 3. It is now set to " + str(window_length))

        self.window_length = int(window_length)
        self.stride = int(stride)
        self.half = self.window_length // 2

        # build prototype Hann window (unscaled)
        w = self._hann(self.window_length)          # shape (window_length,)

        # numerical derivative dw/dt (unscaled)
        dw = self.finite_difference(w, dt=self.dt, axis=-1)

        # normalisation factor so that ∑ phi * dt ≈ 1, like your old alpha
        base_alpha = 1.0 / (np.sum(w) * self.dt)
        total_alpha = self.alpha * base_alpha

        # scaled prototype window and derivative
        self.phi = w * total_alpha        # what WeakFormulation expects
        self.dphi = dw * total_alpha      # what WeakFormulation expects

        # centres will be computed once we know N (grid length)
        self._centers: Array | None = None

    # ------------------------------------------------------------------
    # Internal: Hann shape on [0, window_length-1]
    # ------------------------------------------------------------------
    @staticmethod
    def _hann(L: int) -> Array:
        if L % 2 == 0 or L < 3:
            raise ValueError("Hann length must be an odd integer ≥ 3.")
        n = np.arange(L)
        w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (L - 1))
        w[0] = 0.0
        w[-1] = 0.0
        return w

    # ------------------------------------------------------------------
    # Centres and grid-dependent behaviour
    # ------------------------------------------------------------------
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

    # Optional helper: build all windows explicitly on a grid
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