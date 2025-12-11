from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from collections.abc import Sequence

import numpy as np
import pysindy as ps

from ddmo.utils.data_constructor import DataConstructor
from ddmo.utils.array_helpers import normalize_opt_vars, stack_state_matrix, ensure_callable_name
from .base import Formulation

Builder = Callable[..., Any]


@dataclass
class StrongFormulation(Formulation):
    """Strong formulation built from explicit left- and right-hand builders."""
    
    data_instance: DataConstructor | None = None
    class_attributes: dict[str, Any] | None = None
    lhs_builder: Builder | None = None
    rhs_builder: Builder | None = None

    def get_lhs(self, *args: Any, **kwargs: Any) -> Any:
        if self.lhs_builder is not None:
            return self.lhs_builder(*args, **kwargs)

        data_instance = kwargs.get("data_instance")
        if data_instance is None and args:
            data_instance = args[0]

        if data_instance is None or not hasattr(data_instance, "x_dot"):
            msg = (
                "StrongFormulation requires a data_instance with an x_dot attribute "
                "when no lhs_builder is supplied."
            )
            raise ValueError(msg)
        return data_instance.x_dot

    def get_rhs(self, *args: Any, **kwargs: Any) -> Any:
        if self.rhs_builder is not None:
            return self.rhs_builder(*args, **kwargs)

        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)

        named_funcs = data_instance.library_object([*opt_vars])
        functions = [func for func, _ in named_funcs]
        raw_names = [name for _, name in named_funcs]
        function_names = [ensure_callable_name(name) for name in raw_names]

        x_matrix = stack_state_matrix(data_instance.x_train)

        lib_custom = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names
        )
        lib_custom.fit(x_matrix)
        theta = lib_custom.transform(x_matrix)

        return {
            "theta": theta,
            "library": lib_custom,
            "opt_vars": opt_vars,
            "named_funcs": named_funcs,
        }


def _extract_rhs_inputs(*args: Any, **kwargs: Any) -> tuple[Any, tuple[float, ...]]:
    data_instance = None
    opt_vars = None

    if args:
        data_instance = args[0]
        if len(args) > 1:
            opt_vars = args[1]

    data_instance = kwargs.get("data_instance", data_instance)
    opt_vars = kwargs.get("optimization_variables", opt_vars)

    if data_instance is None:
        msg = "StrongFormulation.get_rhs requires a data_instance."
        raise ValueError(msg)

    if opt_vars is None:
        msg = "StrongFormulation.get_rhs requires optimization variables."
        raise ValueError(msg)

    normalized_opt_vars = normalize_opt_vars(opt_vars)
    return data_instance, normalized_opt_vars
