from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DataConstructor:
    x_train: np.ndarray
    time_array: np.ndarray
    library_object: Any
    x_0: list
    x_raw: np.ndarray
    x_dot: Optional[np.ndarray] = None
    dimension: int = field(init=False)
    number_initial_conditions: int = field(init=False)

    def __post_init__(self):
        self.dimension = self.x_train.shape[1]
        self.number_initial_conditions = len(self.x_0)
