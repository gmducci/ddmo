import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConstructor:
    x_train: np.ndarray
    time_array: np.ndarray
    x_dot: np.ndarray
    library_object: Any
    x_0: list
    x_raw: np.ndarray
    dimension: int = field(init=False)
    number_initial_conditions: int = field(init=False)

    def __post_init__(self):
        self.dimension = self.x_train.shape[1]
        self.number_initial_conditions = len(self.x_0)