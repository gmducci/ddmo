from collections.abc import ABC, abstractmethod
import numpy as np

class FunctionClass(ABC):
    pass


def get_function_class(name: str) -> FunctionClass:
    if name == "HannWindow":
        return HannWindow()
    else:
        raise ValueError(f"Unknown function class: {name}")


funct1 = get_function_class("HannWindow")
funct1.evaluate()


class HannWindow(FunctionClass):

    name= "HannWindow"

    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate(length) -> np.ndarray:
        if length % 2 == 0 or length < 3:
            raise ValueError("Window length must be an odd integer ≥ 3.")

        n = np.arange(length)
        w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (length - 1))
        w[0] = 0.0
        w[-1] = 0.0
        return w

    def __call__(self, length) -> np.ndarray:
        return self.evaluate(length)