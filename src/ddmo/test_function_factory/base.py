# base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Sequence
import numpy as np

Array = np.ndarray


class TestFunctionFamily(ABC):
    """
    Abstract base class for test-function families on a regular grid.

    It does NOT assume any particular parameterisation (window length, stride, etc.).
    Subclasses decide their own parameters, but must be able to:
      • build test functions on a given grid
      • report where those tests are centred.
    """

    name: str = "TestFunctionFamily"

    def __init__(self, dt: float = 1.0, alpha: float = 1.0) -> None:
        self.dt = float(dt)
        self.alpha = float(alpha)
        self._centers: Optional[Array] = None  # shape (K, ndim) or (K,)

    # ---- core abstract interface --------------------------------------

    @abstractmethod
    def build_tests(self, grid_shape: tuple[int, ...]) -> list[Array]:
        """
        Build and return the list of test functions on a grid.

        Parameters
        ----------
        grid_shape : tuple[int, ...]
            Shape of the grid on which data live, e.g.
              • (M,)       for 1D (time or space)
              • (M_x, M_t) for 1D space × time
              • (M_x, M_y) for 2D space, etc.

        Returns
        -------
        list of ndarray
            Each test has shape `grid_shape`.
        """
        ...

    @property
    @abstractmethod
    def centers(self) -> Array:
        """
        Integer grid indices of the centres of each test function.

        Shape (K,) for 1D grids, or (K, ndim) for multi-D.
        """
        ...

    # ---- optional generic helpers -------------------------------------

    @staticmethod
    def finite_difference(phi: Array, dt: float, axis: int = -1) -> Array:
        """
        Central finite-difference derivative along a given axis.

        Endpoints are set to zero; intended as a utility, not a requirement.
        """
        if phi.shape[axis] < 3:
            return np.zeros_like(phi)

        # roll axis to the end
        phi_r = np.moveaxis(phi, axis, -1)
        dphi_r = np.empty_like(phi_r)

        dphi_r[..., 1:-1] = (phi_r[..., 2:] - phi_r[..., :-2]) / (2.0 * dt)
        dphi_r[..., 0] = 0.0
        dphi_r[..., -1] = 0.0

        return np.moveaxis(dphi_r, -1, axis)
    

    @abstractmethod
    def grid_centers(self, M: int) -> Array:
        """Return integer centres of test functions on a 1D grid of length M."""
        pass