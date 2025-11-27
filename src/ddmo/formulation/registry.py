# ddmo/formulation/registry.py
from typing import Type, Any, Dict
from .weak import WeakFormulation
# from .strong_formulation import StrongFormulation
# from .bayesian_formulation import BayesianFormulation

FORMULATION_REGISTRY: Dict[str, Type[Any]] = {
    "weak": WeakFormulation,
    # "strong": StrongFormulation,
    # "bayesian": BayesianFormulation,
}