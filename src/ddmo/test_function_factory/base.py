# ddmo/test_function_factory/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Array = np.ndarray


class TestFunctionFamily(ABC):
    name: str = "TestFunctionFamily"
    allowed_params: set[str] | None = None

    def __init__(
        self,
        dt: float | Sequence[float] = 1.0,
        alpha: float = 1.0,
        params: Mapping[str, Any] | None = None,
        **extra_params: Any,
    ) -> None:
        self.dt = np.array(dt, dtype=float) if np.ndim(dt) else float(dt)
        self.alpha = float(alpha)

        merged: dict[str, Any] = {}
        if params is not None:
            merged.update(params)
        merged.update(extra_params)

        self._validate_param_names(merged)
        self.params: Mapping[str, Any] = merged

        self._centers: Array | None = None

        self._validate_params()

    def _validate_param_names(self, params: Mapping[str, Any]) -> None:
        if self.allowed_params is None:
            return
        unknown = set(params) - self.allowed_params
        if unknown:
            raise ValueError(
                f"{self.__class__.__name__}: unknown parameters {unknown}. "
                f"Allowed: {sorted(self.allowed_params)}"
            )

    def _validate_params(self) -> None:
        return

    @abstractmethod
    def build_tests(self, grid_shape: tuple[int, ...]) -> list[Array]: ...

    @property
    @abstractmethod
    def centers(self) -> Array: ...

    @staticmethod
    def finite_difference(phi: Array, dt: float, axis: int = -1) -> Array:
        if phi.shape[axis] < 3:
            return np.zeros_like(phi)
        phi_r = np.moveaxis(phi, axis, -1)
        dphi_r = np.empty_like(phi_r)
        dphi_r[..., 1:-1] = (phi_r[..., 2:] - phi_r[..., :-2]) / (2.0 * dt)
        dphi_r[..., 0] = 0.0
        dphi_r[..., -1] = 0.0
        return np.moveaxis(dphi_r, -1, axis)
