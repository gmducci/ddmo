from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pysindy as ps

from ddmo.utils.data_constructor import DataConstructor


@dataclass(frozen=True)
class RegressionProblem:
    """Container describing the regression system to be solved."""

    state_matrix: np.ndarray
    feature_matrix: Optional[np.ndarray]
    rhs: Optional[np.ndarray]
    feature_library: Any
    opt_vars: Tuple[float, ...]
    time_vector: Optional[np.ndarray]
    time_step: Optional[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeakLibraryRequest:
    """Data exposed to a user-provided weak-form assembly routine."""

    opt_vars: Tuple[float, ...]
    functions: list
    function_names: list
    state_matrix: np.ndarray
    time_grid: Optional[np.ndarray]
    time_step: Optional[float]
    derivative_matrix: Optional[np.ndarray]
    weak_options: Mapping[str, Any]
    data_instance: DataConstructor


WeakLibraryFactory = Callable[[WeakLibraryRequest], RegressionProblem]


class CachedFeatureLibrary:
    """Wrapper that reuses a precomputed feature matrix for the training data."""

    def __init__(
        self,
        base_library: Any,
        theta_matrix: np.ndarray,
        training_matrix: np.ndarray,
    ):
        self._base_library = base_library
        self._theta = theta_matrix
        self._training_matrix = training_matrix

    def fit(self, x, y=None):
        self._base_library.fit(x, y)
        return self

    def transform(self, x):
        if self._is_training_input(x):
            return self._theta
        return self._base_library.transform(x)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def get_feature_names(self, input_features=None):
        return self._base_library.get_feature_names(input_features)

    def _is_training_input(self, x) -> bool:
        if x is self._training_matrix:
            return True
        try:
            return (
                x.shape == self._training_matrix.shape
                and np.shares_memory(x, self._training_matrix)
                and np.array_equal(x, self._training_matrix)
            )
        except Exception:
            return False

    def __getattr__(self, item):
        return getattr(self._base_library, item)

    def __setattr__(self, key, value):
        if key in {"_base_library", "_theta", "_training_matrix"}:
            super().__setattr__(key, value)
        else:
            setattr(self._base_library, key, value)


class FeatureMatrixBuilder:
    """Builds the regression problem used by sparse regression solvers."""

    def __init__(
        self,
        data_instance: DataConstructor,
        *,
        formulation: str = "strong",
        weak_options: Optional[Mapping[str, Any]] = None,
        weak_library_factory: Optional[WeakLibraryFactory] = None,
    ):
        self.data_instance = data_instance
        self.formulation = formulation.lower()
        self.weak_options: dict[str, Any] = dict(weak_options or {})
        self.weak_library_factory = weak_library_factory
        self._validate_formulation()

    def build_problem(self, optimization_vars: Sequence[float] | float) -> RegressionProblem:
        """Assemble the regression problem (LHS/RHS) for the given hyper-parameters."""
        opt_vars = self._normalize_optimization_vars(optimization_vars)
        state_matrix = self._build_state_matrix()
        raw_time_grid = self._compute_time_grid()
        time_grid, time_step = self._align_time_information(raw_time_grid, state_matrix)
        derivative_matrix = self._build_derivative_matrix()

        if self.formulation == "weak":
            return self._build_weak_problem(
                opt_vars,
                state_matrix,
                time_grid,
                time_step,
                derivative_matrix,
            )

        return self._build_strong_problem(
            opt_vars,
            state_matrix,
            time_grid,
            time_step,
            derivative_matrix,
        )

    def _build_strong_problem(
        self,
        opt_vars: Tuple[float, ...],
        state_matrix: np.ndarray,
        time_grid: Optional[np.ndarray],
        time_step: Optional[float],
        derivative_matrix: Optional[np.ndarray],
    ) -> RegressionProblem:
        if derivative_matrix is None:
            raise ValueError(
                "Strong formulation requires training derivatives (`x_dot`) in the data instance."
            )

        base_library = self._build_strong_library(opt_vars, state_matrix)
        theta_matrix = base_library.transform(state_matrix)
        cached_library = CachedFeatureLibrary(base_library, theta_matrix, state_matrix)

        time_vector = (
            time_grid if time_grid is not None and time_grid.ndim == 1 and time_grid.shape[0] == state_matrix.shape[0]
            else None
        )

        return RegressionProblem(
            state_matrix=state_matrix,
            feature_matrix=theta_matrix,
            rhs=derivative_matrix,
            feature_library=cached_library,
            opt_vars=opt_vars,
            time_vector=time_vector,
            time_step=time_step,
            metadata={"base_library": base_library},
        )

    def _build_weak_problem(
        self,
        opt_vars: Tuple[float, ...],
        state_matrix: np.ndarray,
        time_grid: Optional[np.ndarray],
        time_step: Optional[float],
        derivative_matrix: Optional[np.ndarray],
    ) -> RegressionProblem:
        if self.weak_library_factory is None:
            raise NotImplementedError(
                "Weak formulation requested, but no `weak_library_factory` was provided. "
                "Supply a callable when creating FeatureMatrixBuilder or via solver configuration."
            )

        functions, function_names = self._extract_library_components(opt_vars)
        request = WeakLibraryRequest(
            opt_vars=opt_vars,
            functions=functions,
            function_names=function_names,
            state_matrix=state_matrix,
            time_grid=time_grid,
            time_step=time_step,
            derivative_matrix=derivative_matrix,
            weak_options=self.weak_options,
            data_instance=self.data_instance,
        )

        problem = self.weak_library_factory(request)
        if not isinstance(problem, RegressionProblem):
            raise TypeError(
                "Weak library factory must return a `RegressionProblem` instance."
            )
        if problem.state_matrix is None:
            raise ValueError("Weak library factory must provide the state matrix used for regression.")
        return problem

    def _build_strong_library(self, opt_vars: Tuple[float, ...], state_matrix: np.ndarray) -> ps.CustomLibrary:
        functions, function_names = self._extract_library_components(opt_vars)
        library = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names,
        )
        library.fit(state_matrix)
        return library

    def _extract_library_components(self, opt_vars: Tuple[float, ...]) -> Tuple[list, list]:
        named_funcs = self.data_instance.library_object(list(opt_vars))
        functions = [func for func, _ in named_funcs]
        function_names = [name for _, name in named_funcs]
        return functions, function_names

    def _build_state_matrix(self) -> np.ndarray:
        x_train = np.asarray(self.data_instance.x_train)
        if x_train.ndim > 2:
            # Collapse multiple trajectories while keeping variable dimension.
            return x_train.reshape(-1, x_train.shape[-1])
        return x_train

    def _build_derivative_matrix(self) -> Optional[np.ndarray]:
        x_dot = getattr(self.data_instance, "x_dot", None)
        if x_dot is None:
            return None
        x_dot = np.asarray(x_dot)
        if x_dot.ndim > 2:
            return x_dot.reshape(-1, x_dot.shape[-1])
        return x_dot

    def _compute_time_grid(self) -> Optional[np.ndarray]:
        time_array = getattr(self.data_instance, "time_array", None)
        if time_array is None:
            return None
        return np.asarray(time_array)

    def _align_time_information(
        self,
        time_grid: Optional[np.ndarray],
        state_matrix: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Match time information to the state matrix shape."""
        if time_grid is None:
            return None, None

        time_array = np.asarray(time_grid)
        n_samples = state_matrix.shape[0]

        if time_array.ndim == 1:
            if time_array.shape[0] == n_samples:
                return time_array, self._compute_time_step_from_vector(time_array)
            return None, self._compute_time_step_from_vector(time_array)

        if time_array.shape[0] == n_samples:
            flattened = time_array.reshape(time_array.shape[0], -1)[:, 0]
            return flattened, self._compute_time_step_from_vector(flattened)

        return None, self._compute_time_step_from_vector(time_array.reshape(-1))

    def _compute_time_step_from_vector(self, time_vector: np.ndarray) -> Optional[float]:
        if time_vector is None or time_vector.size < 2:
            return None
        diffs = np.diff(time_vector)
        if np.allclose(diffs, diffs[0]):
            return float(diffs[0])
        return None

    def _normalize_optimization_vars(
        self, optimization_vars: Sequence[float] | np.ndarray | float
    ) -> Tuple[float, ...]:
        """Ensure the optimizer variables are represented as a tuple of floats."""
        if np.isscalar(optimization_vars):
            return (float(optimization_vars),)

        array_like = np.atleast_1d(np.asarray(optimization_vars, dtype=float))
        return tuple(float(v) for v in array_like)

    def _validate_formulation(self) -> None:
        if self.formulation not in {"strong", "weak"}:
            raise ValueError(
                f"Unknown formulation '{self.formulation}'. Supported options are 'strong' and 'weak'."
            )
