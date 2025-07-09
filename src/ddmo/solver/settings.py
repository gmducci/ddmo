from dataclasses import dataclass, field
from typing import Type, Any
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference

@dataclass
class SolverSettings:
    optimizer_cls: Type = STLSQ
    optimizer_params: dict = field(default_factory=dict)

    library_cls: Type = PolynomialLibrary
    library_params: dict = field(default_factory=dict)

    differentiation_cls: Type = FiniteDifference
    differentiation_params: dict = field(default_factory=lambda: {"axis": -2})

    t_default: float = 1.0
    discrete_time: bool = False

    def build(self):
        optimizer = self.optimizer_cls(**self.optimizer_params)
        library = self.library_cls(**self.library_params)
        differentiation = self.differentiation_cls(**self.differentiation_params)

        import pysindy as ps
        return ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            differentiation_method=differentiation,
            t_default=self.t_default,
            discrete_time=self.discrete_time
        )