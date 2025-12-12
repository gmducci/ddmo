# ddmo/formulation/registry.py
from typing import Any

from .strong import StrongFormulation
from .weak import WeakFormulation

# from .bayesian_formulation import BayesianFormulation

FORMULATION_REGISTRY: dict[str, type[Any]] = {
    "weak": WeakFormulation,
    "strong": StrongFormulation,
    # "bayesian": BayesianFormulation,
}
