"""Utilities for generating synthetic ODE trajectory batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint

from ddmo.utils.data_constructor import DataConstructor


ODEFunction = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class ODEDataBatch:
    """Generated ODE trajectories and metadata."""

    time: np.ndarray
    x_clean: np.ndarray
    x_noisy: np.ndarray
    x_dot_clean: np.ndarray
    initial_conditions: list[np.ndarray]

    @property
    def x_train(self) -> np.ndarray:
        """Noisy trajectories ready for model identification."""
        return self.x_noisy

    def to_data_constructor(self, *, include_derivatives: bool = False) -> DataConstructor:
        """Build a DDMO ``DataConstructor`` from the generated trajectories."""
        x_dot = self.x_dot_clean if include_derivatives else None
        return DataConstructor(
            x_train=self.x_train,
            time_array=self.time,
            x_dot=x_dot,
            x_0=self.initial_conditions,
            x_raw=self.x_noisy,
        )


class ODEDataGenerator:
    """Generate batched synthetic data from a generic ODE right-hand side."""

    def __init__(
        self,
        ode: ODEFunction,
        *,
        initial_conditions: Sequence[Sequence[float] | np.ndarray],
        t_start: float = 0.0,
        t_end: float = 1.0,
        sampling_frequency: float | None = None,
        time_array: Sequence[float] | np.ndarray | None = None,
        noise_level: float = 0.0,
        seed: int | None = None,
        clip_min: float | None = None,
    ):
        """Configure an ODE trajectory generator.

        Parameters
        ----------
        ode:
            Callable with the ``odeint`` signature ``f(state, time)``.
        initial_conditions:
            Batch of initial states. Each item produces one trajectory.
        t_start, t_end, sampling_frequency:
            Time-grid settings used when ``time_array`` is not provided.
        time_array:
            Optional explicit time vector. If supplied, it takes precedence over
            ``t_start``, ``t_end``, and ``sampling_frequency``.
        noise_level:
            Gaussian noise multiplier relative to each clean state variable's
            empirical standard deviation.
        seed:
            Optional random seed for reproducible noise.
        clip_min:
            Optional lower bound applied after adding noise. This is useful for
            ODEs whose candidate libraries require positive states.
        """
        self.ode = ode
        self.initial_conditions = [
            np.asarray(x0, dtype=float) for x0 in initial_conditions
        ]
        self.t_start = t_start
        self.t_end = t_end
        self.sampling_frequency = sampling_frequency
        self.time_array = (
            None if time_array is None else np.asarray(time_array, dtype=float)
        )
        self.noise_level = noise_level
        self.seed = seed
        self.clip_min = clip_min
        self._validate()

    def generate(self) -> ODEDataBatch:
        """Integrate the ODE for every initial condition and add optional noise."""
        rng = np.random.default_rng(self.seed)
        time = self._make_time_array()
        clean_trajectories = []
        noisy_trajectories = []
        derivative_trajectories = []

        for x0 in self.initial_conditions:
            x_clean = odeint(self.ode, x0, time)
            x_dot_clean = np.asarray(
                [self.ode(state, t) for state, t in zip(x_clean, time)]
            )
            x_noisy = self._add_noise(x_clean, rng)

            clean_trajectories.append(x_clean)
            noisy_trajectories.append(x_noisy)
            derivative_trajectories.append(x_dot_clean)

        return ODEDataBatch(
            time=time,
            x_clean=np.stack(clean_trajectories, axis=0),
            x_noisy=np.stack(noisy_trajectories, axis=0),
            x_dot_clean=np.stack(derivative_trajectories, axis=0),
            initial_conditions=[x0.copy() for x0 in self.initial_conditions],
        )

    def _validate(self) -> None:
        if not callable(self.ode):
            raise ValueError("ode must be callable.")
        if not self.initial_conditions:
            raise ValueError("initial_conditions must contain at least one state.")
        dimension = self.initial_conditions[0].shape
        if any(x0.shape != dimension for x0 in self.initial_conditions):
            raise ValueError("all initial conditions must have the same shape.")
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative.")
        if self.time_array is None:
            if self.sampling_frequency is None:
                raise ValueError(
                    "sampling_frequency is required when time_array is not provided."
                )
            if self.sampling_frequency <= 0:
                raise ValueError("sampling_frequency must be positive.")
            if self.t_end <= self.t_start:
                raise ValueError("t_end must be greater than t_start.")
        else:
            if self.time_array.ndim != 1:
                raise ValueError("time_array must be one-dimensional.")
            if len(self.time_array) < 2:
                raise ValueError("time_array must contain at least two points.")
            if np.any(np.diff(self.time_array) <= 0):
                raise ValueError("time_array must be strictly increasing.")

    def _make_time_array(self) -> np.ndarray:
        if self.time_array is not None:
            return self.time_array.copy()

        dt = 1.0 / float(self.sampling_frequency)
        return np.arange(self.t_start, self.t_end + 0.5 * dt, dt)

    def _add_noise(
        self,
        x_clean: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.noise_level == 0:
            x_noisy = x_clean.copy()
        else:
            scale = np.std(x_clean, axis=0)
            noise = rng.normal(0.0, self.noise_level, size=x_clean.shape) * scale
            x_noisy = x_clean + noise

        if self.clip_min is not None:
            x_noisy = np.clip(x_noisy, self.clip_min, None)

        return x_noisy
