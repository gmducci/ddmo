from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pysindy as ps

from ddmo.formulation.base import Formulation
from ddmo.test_function_factory.base import TestFunctionFamily  # abstract base
from ddmo.test_function_factory.registry import WEAK_TEST_FAMILY_REGISTRY
from ddmo.utils.array_helpers import (
    ensure_callable_name,
    normalize_opt_vars,
    stack_state_matrix,
)
from ddmo.utils.data_constructor import DataConstructor

IntegralSolver = Callable[..., np.ndarray]
FamilyBuilder = Callable[[dict[str, float]], TestFunctionFamily]  # <- dict, not tuple

Array = np.ndarray


@dataclass
class WeakFormulation(Formulation):
    """
    Weak formulation driven by a test-function family chosen from a registry.

    Parameters
    ----------
    data_instance: DataConstructor
        Object containing time_array and x_train.
    class_attributes : dict
        Dictionary that must at least contain:
            - "test_functions_class": name in WEAK_TEST_FAMILY_REGISTRY
        and may contain any other keyword arguments for that test-family
        (e.g. "window_length", "stride", ...).
    """

    data_instance: DataConstructor
    class_attributes: dict[str, Any]
    library_object: Callable | None = None
    integral_solver: IntegralSolver | None = None
    family_builder: FamilyBuilder | None = None

    def __post_init__(self) -> None:
        self.time_array = self.data_instance.time_array
        self.x_train = self.data_instance.x_train

        self.x_train_blocks = _as_trajectory_blocks(self.x_train)

        # Basic shape checks
        if self.time_array.ndim != 1:
            raise ValueError("time_array must be 1-D.")
        if self.time_array.size < 2 or not np.all(np.isfinite(self.time_array)):
            raise ValueError("time_array must contain at least two finite points.")
        if any(
            block.shape[0] != self.time_array.shape[0] for block in self.x_train_blocks
        ):
            raise ValueError(
                "Each trajectory in x_train must have the same length as time_array."
            )

        var_counts = {block.shape[1] for block in self.x_train_blocks}
        if len(var_counts) != 1:
            raise ValueError("All trajectories in x_train must have the same number of variables.")

        steps = np.diff(self.time_array)
        self.dt = float(steps[0])
        if self.dt <= 0 or not np.allclose(
            steps, self.dt, rtol=1e-10, atol=1e-12 * self.dt
        ):
            raise ValueError("WeakFormulation requires a strictly increasing uniform time_array.")
        self.N = self.x_train_blocks[0].shape[0]
        self.var_numbers = self.x_train_blocks[0].shape[1]

        # ------------------------------------------------------------------
        # Build initial test-family from the registry
        # ------------------------------------------------------------------
        family_name = self.class_attributes.get("test_functions_class")
        if family_name is None:
            raise ValueError(
                "class_attributes must contain key 'test_functions_class'."
            )

        try:
            family_cls = WEAK_TEST_FAMILY_REGISTRY[family_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown test-function family '{family_name}'. "
                f"Available: {sorted(WEAK_TEST_FAMILY_REGISTRY)}"
            ) from exc

        # All other entries in class_attributes are treated as parameters
        base_params = {
            k: v
            for k, v in self.class_attributes.items()
            if k != "test_functions_class"
        }

        # Build the initial family and configure the formulation
        initial_family = family_cls(dt=self.dt, **base_params)
        self._configure_from_family(initial_family)

        # ------------------------------------------------------------------
        # Build a family_builder closure for later updates (GA, etc.)
        # ------------------------------------------------------------------
        def builder(param_dict: dict[str, float]) -> TestFunctionFamily:
            """
            Create a new test-family instance from updated optimisation parameters.

            param_dict maps parameter names (e.g. 'gamma_1', 'stride') to floats.
            """
            params = dict(base_params)

            for name, val in param_dict.items():
                if name not in params:
                    # ignore optimisation variables that do not correspond
                    # to test-family parameters
                    continue

                if name == "stride":
                    params[name] = int(round(val))
                elif name == "window_length":
                    params[name] = int(round(val)) * 2 + 1
                else:
                    params[name] = float(val)

            return family_cls(dt=self.dt, **params)

        self.family_builder = builder

    # ------------------------------------------------------------------
    # Configuration from a TestFunctionFamily
    # ------------------------------------------------------------------
    def _configure_from_family(self, family: TestFunctionFamily) -> None:
        self.test_family = family

        try:
            phi = np.asarray(family.phi, float)
            dphi = np.asarray(family.dphi, float)
        except AttributeError as exc:
            raise AttributeError(
                "test_family must provide `phi` and `dphi` attributes "
                "(prototype test function and its derivative)."
            ) from exc

        if phi.shape != dphi.shape or phi.ndim != 1:
            raise ValueError(
                "test_family.phi and test_family.dphi must be 1-D arrays "
                "of the same length."
            )
        if phi.size < 2 or not np.all(np.isfinite(phi)):
            raise ValueError("test_family.phi must contain at least two finite samples.")

        window_length = phi.shape[0]

        self.window_length = window_length
        self.phi = phi
        self.dphi = dphi
        self.half = window_length // 2
        # The same piecewise-linear phi defines both sides of the discrete
        # weak identity. With linearly interpolated state data these weights
        # integrate x_dot * phi and f(x) * phi exactly on each grid interval.
        self.w_lhs, self.w_rhs = _build_consistent_weak_weights(self.phi, self.dt)

        try:
            centers_by_trajectory = [
                np.asarray(family.grid_centers(block.shape[0]), dtype=int)
                for block in self.x_train_blocks
            ]
        except AttributeError as exc:
            raise AttributeError(
                "test_family must implement `grid_centers(M: int) -> np.ndarray` "
                "to provide window centres."
            ) from exc

        self.centers_by_trajectory = centers_by_trajectory
        self.centers = (
            np.concatenate(centers_by_trajectory)
            if centers_by_trajectory
            else np.array([], dtype=int)
        )
        self.M = sum(centers.shape[0] for centers in centers_by_trajectory)

    # ------------------------------------------------------------------
    # Left-hand side (unchanged integral logic)
    # ------------------------------------------------------------------
    def get_lhs(self, *args: Any, **kwargs: Any) -> np.ndarray:
        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)

        param_dict = kwargs.get("opt_param_dict", None)
        if self.family_builder is not None and param_dict is not None:
            new_family = self.family_builder(param_dict)
            self._configure_from_family(new_family)

        blocks = _as_trajectory_blocks(data_instance.x_train)
        lhs_blocks = []

        for block, centers in zip(blocks, self.centers_by_trajectory):
            B = np.zeros((centers.shape[0], self.var_numbers))
            for i in range(self.var_numbers):
                B[:, i] = _sliding_integral(block[:, i], centers, self.w_lhs)
            lhs_blocks.append(B)

        return np.vstack(lhs_blocks)

    # ------------------------------------------------------------------
    # Right-hand side (unchanged integral logic)
    # ------------------------------------------------------------------
    def get_rhs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data_instance, opt_vars = _extract_rhs_inputs(*args, **kwargs)

        param_dict = kwargs.get("opt_param_dict", None)
        if self.family_builder is not None and param_dict is not None:
            new_family = self.family_builder(param_dict)
            self._configure_from_family(new_family)

        # Pass dict to library function (safer than positional list)
        # If param_dict not available, fall back to list for backward compat
        library_object = kwargs.get("library_object", self.library_object)
        if library_object is None:
            msg = "WeakFormulation.get_rhs requires a candidate library callable."
            raise ValueError(msg)
        if param_dict is not None:
            named_funcs = library_object(param_dict)
        else:
            named_funcs = library_object([*opt_vars])
        functions = [func for func, _ in named_funcs]
        raw_names = [name for _, name in named_funcs]
        function_names = [ensure_callable_name(name) for name in raw_names]

        trajectory_blocks = _as_trajectory_blocks(data_instance.x_train)
        x_matrix = stack_state_matrix(trajectory_blocks)

        lib_custom = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names,
        )
        lib_custom.fit(x_matrix)
        theta = lib_custom.transform(x_matrix)
        colmask = ~np.isnan(theta).any(axis=0)

        theta = theta[:, colmask]
        theta_blocks = _split_stacked_matrix(theta, trajectory_blocks)
        rhs_blocks = []

        for theta_block, centers in zip(theta_blocks, self.centers_by_trajectory):
            rhs = np.zeros((centers.shape[0], theta.shape[1]))
            for j, G in enumerate(theta_block.T):
                rhs[:, j] = _sliding_integral(G, centers, self.w_rhs)
            rhs_blocks.append(rhs)

        return {
            "theta": np.vstack(rhs_blocks),
            "library": lib_custom,
            "opt_vars": opt_vars,
            "named_funcs": named_funcs,
            "colmask": colmask,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_trajectory_blocks(values: Any) -> list[np.ndarray]:
    """Normalize trajectory data to a list of 2-D arrays."""
    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            return [np.asarray(values)]
        if values.ndim == 3:
            return [np.asarray(values[i]) for i in range(values.shape[0])]
        raise ValueError("x_train must be 2-D or 3-D for weak formulation.")

    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        blocks = [np.asarray(block) for block in values]
        if not blocks:
            raise ValueError("x_train must contain at least one trajectory.")
        if any(block.ndim != 2 for block in blocks):
            raise ValueError("Each trajectory in x_train must be 2-D.")
        return blocks

    array = np.asarray(values)
    if array.ndim == 2:
        return [array]
    raise ValueError("x_train must be 2-D or a sequence of 2-D trajectories.")


def _split_stacked_matrix(
    stacked: np.ndarray,
    trajectory_blocks: list[np.ndarray],
) -> list[np.ndarray]:
    lengths = [block.shape[0] for block in trajectory_blocks]
    split_points = np.cumsum(lengths)[:-1]
    return list(np.split(stacked, split_points, axis=0))


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
        msg = "WeakFormulation.get_rhs requires a data_instance."
        raise ValueError(msg)

    if opt_vars is None:
        msg = "WeakFormulation.get_rhs requires optimization variables."
        raise ValueError(msg)

    normalized_opt_vars = normalize_opt_vars(opt_vars)
    return data_instance, normalized_opt_vars


def _sliding_integral(
    signal: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute windowed integrals using precomputed weights.

    signal  : shape (N,)
    centers : shape (M,), integer indices of window centres
    weights : shape (L,), precomputed weights on a reference window

    Returns
    -------
    out : shape (M,)
        out[m] = sum_{k=0}^{L-1} signal[c_m - h + k] * weights[k]
    """
    L = weights.shape[0]
    h = L // 2
    out = np.empty(centers.shape[0], dtype=float)

    for m, c in enumerate(centers):
        sl = slice(c - h, c + h + 1)
        out[m] = np.dot(signal[sl], weights)

    return out


def _build_consistent_weak_weights(
    phi: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nodal weights for linear state and test-function interpolants.

    The LHS represents ∫ x_dot phi dt. Equivalently it includes the boundary
    term [x phi] - ∫ x phi_dot dt, even when phi does not vanish at the ends.
    The RHS represents ∫ f phi dt with linearly interpolated nodal f values.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or phi.size < 2 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must contain at least two finite samples.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    lhs = np.zeros_like(phi)
    rhs = np.zeros_like(phi)
    area = (phi[:-1] + phi[1:]) / 2
    lhs[:-1] -= area
    lhs[1:] += area
    rhs[:-1] += dt * (2 * phi[:-1] + phi[1:]) / 6
    rhs[1:] += dt * (phi[:-1] + 2 * phi[1:]) / 6
    return lhs, rhs
