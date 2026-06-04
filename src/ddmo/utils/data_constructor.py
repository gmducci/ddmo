from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DataConstructor:
    x_train: np.ndarray
    time_array: np.ndarray
    x_0: list
    x_raw: np.ndarray
    x_dot: Optional[np.ndarray] = None
    dimension: int = field(init=False)
    number_initial_conditions: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.x_train, np.ndarray):
            self.dimension = self.x_train.shape[-1]
        elif isinstance(self.x_train, Sequence) and not isinstance(
            self.x_train, (str, bytes)
        ):
            self.dimension = np.asarray(self.x_train[0]).shape[-1]
        else:
            self.dimension = np.asarray(self.x_train).shape[-1]
        self.number_initial_conditions = len(self.x_0)
