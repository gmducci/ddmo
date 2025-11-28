# ddmo/formulation/formulation_settings.py
from dataclasses import dataclass, field
from typing import Mapping, Any


@dataclass
class FormulationSettings:
    """
    Generic settings for a formulation.

    Attributes
    ----------
    formulation_name : str
        Name used in the global FORMULATION_REGISTRY.
    formulation_parameters : Mapping[str, Any]
        Arbitrary kwargs passed to the formulation (interpreted by that formulation).
    """
    formulation_name: str
    formulation_attributes: Mapping[str, Any] = field(default_factory=dict)