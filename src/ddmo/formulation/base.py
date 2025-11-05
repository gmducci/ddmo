from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Formulation(ABC):
    """Shared interface for problem formulations (e.g., weak, strong)."""

    @abstractmethod
    def get_lhs(self, *args: Any, **kwargs: Any) -> Any:
        """Return the contribution for the left-hand side of the system."""
        raise NotImplementedError

    @abstractmethod
    def get_rhs(self, *args: Any, **kwargs: Any) -> Any:
        """Return the contribution for the right-hand side of the system."""
        raise NotImplementedError
