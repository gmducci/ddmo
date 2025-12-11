# ddmo/test_function_factory/polynomial_window.py

from __future__ import annotations

import numpy as np

from .base import Array, TestFunctionFamily


class PolynomialWindow(TestFunctionFamily):
    """
    Polynomial-window test-function family on a 1D time grid.

    Parameters
    ----------
    window_length : int
        Length of each window (odd integer ≥ 3), in grid points.
    stride : int
        Step between consecutive window centres, in grid points.
    degree : int
        Degree of the polynomial bump (>= 0).
    dt : float, optional
        Time step of the data.
    alpha : float, optional
        Extra scaling factor. Effective scaling is chosen so that
        sum(phi) * dt ≈ 1, then multiplied by alpha.
    """

    name = "PolynomialWindow"

    def __init__(
        self,
        window_length: int,
        stride: int,
        degree: int,
        dt: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(dt=dt, alpha=alpha)

        if window_length % 2 == 0 or window_length < 3:
            raise ValueError(
                f"window_length must be an odd integer ≥ 3, got {window_length}."
            )
        if degree < 0:
            raise ValueError(f"degree must be ≥ 0, got {degree}.")

        self.window_length = int(window_length)
        self.stride = int(stride)
        self.degree = int(degree)
        self.half = self.window_length // 2

        # build prototype polynomial bump (unscaled)
        w = self._poly_bump(self.window_length, self.degree)  # shape (L,)

        # numerical derivative dw/dt (unscaled)
        dw = self.finite_difference(w, dt=self.dt, axis=-1)

        # normalisation so that ∑ phi * dt ≈ 1
        base_alpha = 1.0 / (np.sum(w) * self.dt)
        total_alpha = self.alpha * base_alpha

        self.phi = w * total_alpha
        self.dphi = dw * total_alpha

        self._centers: Array | None = None

    @staticmethod
    def _poly_bump(L: int, degree: int) -> Array:
        """
        Symmetric polynomial bump on [-1, 1], sampled on L points, with
        zero at the boundary. Shape ~ (1 - |x|)^degree for |x| <= 1.
        """
        if L % 2 == 0 or L < 3:
            raise ValueError("Polynomial length must be an odd integer ≥ 3.")

        x = np.linspace(-1.0, 1.0, L)  # symmetric grid
        base = 1.0 - np.abs(x)  # in [0, 1], zero at ends
        base[base < 0.0] = 0.0
        w = base**degree if degree > 0 else np.ones_like(base)
        # enforce exact zeros at the boundaries
        w[0] = 0.0
        w[-1] = 0.0
        return w

    # ---- centres / grid-dependent behaviour (same logic as Hann) ----

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
                f"PolynomialWindow.build_tests only supports 1D grids, got {grid_shape}."
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
