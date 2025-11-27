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
    
    def build_formulation(self, data_instance):
        from ddmo.formulation.weak import WeakFormulation

        dt = float(data_instance.time_array[1] - data_instance.time_array[0])

        if self.formulation_name == "weak":
            family_cls = self.test_functions_class
            if family_cls is None:
                raise ValueError("Weak formulation requires a test_functions_class.")

            base_params = dict(self.test_function_parameters)
            test_family = family_cls(dt=dt, **base_params)

            def family_builder(param_dict: dict[str, float]):
                """
                Update test-function parameters from *named* optimisation variables.
                Uses formulation_opt_param_names as a filter.
                """
                params = dict(base_params)

                names = self.formulation_opt_param_names or []
                for name in names:
                    if name in param_dict:
                        val = param_dict[name]
                        if name == "stride":
                            params[name] = int(round(val))
                        else:
                            params[name] = float(val)

                return family_cls(dt=dt, **params)

            return WeakFormulation(
                time_array=data_instance.time_array,
                x_train=data_instance.x_train,
                test_family=test_family,
                family_builder=family_builder,
            )

        elif self.formulation_name == "strong":
            raise NotImplementedError("Strong formulation construction not implemented yet.")
    # Optional convenience constructors
    @classmethod
    def strong(cls, **formulation_parameters: Any) -> "FormulationSettings":
        return cls(
            formulation_name="strong",
            test_functions_class=None,
            formulation_parameters=None,
        )

    @classmethod
    def weak(
        cls,
        test_functions_class: Any,
        **formulation_parameters: Any,
    ) -> "FormulationSettings":
        return cls(
            formulation_name="weak",
            test_functions_class=test_functions_class,
            test_function_parameters=formulation_parameters,
        )