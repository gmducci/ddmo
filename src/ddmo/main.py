#!/usr/bin/env python3
"""
DDMO: One-stop sparse identification entry point

Usage:

  # As a library:
  >>> from ddmo import identify
  >>> from ddmo.utils.constructors import OptimizerSettings
  >>>
  >>> optimizer_settings = OptimizerSettings(
  ...     library_bounds={"gamma_1": (1.0, 10.0)},
  ...     formulation_bounds={},
  ...     n_gen=100,
  ...     algorithm="NSGA2",
  ...     pop_size=50
  ... )
  >>> result = identify(
  ...     data_instance=training_data,
  ...     optimizer_settings=optimizer_settings,
  ...     solver_settings=solver_settings,
  ...     formulation_settings=formulation_settings,
  ...     library_object=library.build
  ... )
"""

import logging
import multiprocessing as mp
import platform
from collections.abc import Callable, Mapping

if mp.current_process().name == "MainProcess":
    _method = "fork" if platform.system() == "Linux" else "spawn"
    try:
        mp.set_start_method(_method, force=False)
    except RuntimeError:
        pass  # Already set by the caller

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from ddmo.formulation.formulation_settings import FormulationSettings
from ddmo.formulation.registry import FORMULATION_REGISTRY
from ddmo.solver.registry import SOLVER_REGISTRY
from ddmo.solver.settings import SolverSettings
from ddmo.utils.constructors import IdentificationResult, OptimizerSettings
from ddmo.utils.data_constructor import DataConstructor

logger = logging.getLogger(__name__)
# Algorithm registry for multi-objective optimization
ALGORITHM_REGISTRY = {"NSGA2": NSGA2, "NSGA3": NSGA3}

# Type aliases for clarity
Func = Callable[..., float]
NamedFunc = tuple[Func, str]


# -----------------------------------------------------------------------------
# Main identification function
# -----------------------------------------------------------------------------


def identify(
    data_instance: DataConstructor,
    optimizer_settings: OptimizerSettings,
    solver_settings: SolverSettings,
    formulation_settings: FormulationSettings | None = None,
    library_object: Callable | None = None,
    objective_functions: Mapping[str, Callable] | None = None,
):
    """
    Run sparse identification on provided data x(t).

    Args:
        data_instance: DataConstructor with training data and library.
        optimizer_settings: Unified optimization configuration.
        formulation_settings: Formulation configuration (e.g., weak, strong).
        library_object: Callable that builds candidate library functions from optimizer parameters.
        objective_functions: Optional mapping of objective names to callables.
            Each callable receives an ObjectiveContext. The objective count is
            derived automatically from this mapping.

    Returns:
        IdentificationResult: Structured result containing problem, result,
            objective_space, solutions, coeff_matrices, alphas, models, theta_matrices.
    """
    from ddmo.optimizer.optimizer import Optimizer

    if formulation_settings is None:
        raise ValueError(
            "identify requires a formulation instance (FormulationSettings)."
        )
    if library_object is None:
        raise ValueError(
            "identify requires a candidate library callable (`library_object`)."
        )

    # Instantiate Solver
    solver_class = SOLVER_REGISTRY[solver_settings.solver_name]
    solver = solver_class(
        data_instance=data_instance,
        optimizer=solver_settings.regularization,
        solver_parameters=solver_settings.regularization_parameters,
        scale_features=solver_settings.scale_features,
        ensemble_settings=solver_settings.ensemble,
        library_object=library_object,
    )

    # Instantiate Formulation
    formulation_class = FORMULATION_REGISTRY[formulation_settings.formulation]
    logger.info("Using formulation: %s", formulation_settings.formulation)

    resolved_formulation = formulation_class(
        data_instance=data_instance,
        class_attributes=formulation_settings.to_attributes_dict(),
        library_object=library_object,
    )

    # Build the optimization problem
    problem = Optimizer(
        data_instance=data_instance,
        solver=solver.solve,
        objective_functions=objective_functions,
        formulation=resolved_formulation,
        solver_param_names=optimizer_settings.solver_param_names,
    )

    # Set up variable mappings from bounds
    map_label = {i: n for i, n in enumerate(optimizer_settings.xl)}
    problem.map_label = map_label
    mapi_label = {n: i for i, n in enumerate(optimizer_settings.xl)}
    problem.mapi_label = mapi_label

    # Store library param names for dict-based access in formulations
    problem.library_param_names = optimizer_settings.library_param_names

    # Convert bounds dictionaries to arrays
    xl = np.array(
        [optimizer_settings.xl[key] for key in optimizer_settings.xl], dtype=float
    )
    xu = np.array(
        [optimizer_settings.xu[key] for key in optimizer_settings.xu], dtype=float
    )

    problem.n_var = optimizer_settings.n_var
    if (
        optimizer_settings.n_obj is not None
        and optimizer_settings.n_obj != problem.n_obj
    ):
        logger.warning(
            "Ignoring n_obj=%s because %s objective functions were configured.",
            optimizer_settings.n_obj,
            problem.n_obj,
        )
    logger.info("Bounds: xl=%s, xu=%s", xl, xu)
    problem.xl = xl
    problem.xu = xu

    # Build pymoo operators from settings
    sampling = FloatRandomSampling()
    crossover = SBX(
        eta=optimizer_settings.crossover_eta, prob=optimizer_settings.crossover_prob
    )
    mutation = PM(eta=optimizer_settings.mutation_eta)

    # Select algorithm from registry
    algorithm_name = optimizer_settings.algorithm
    if algorithm_name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )

    algorithm_class = ALGORITHM_REGISTRY[algorithm_name]

    # Build algorithm kwargs
    algorithm_kwargs = {
        "pop_size": optimizer_settings.pop_size,
        "sampling": sampling,
        "crossover": crossover,
        "mutation": mutation,
        "eliminate_duplicates": True,
    }

    # NSGA3 requires reference directions
    if algorithm_name == "NSGA3":
        ref_dirs = get_reference_directions(
            "das-dennis", problem.n_obj, n_partitions=12
        )
        algorithm_kwargs["ref_dirs"] = ref_dirs

    algorithm = algorithm_class(**algorithm_kwargs)
    logger.info("Using algorithm: %s", algorithm_name)

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', optimizer_settings.n_gen),
        seed=optimizer_settings.seed,
        verbose=optimizer_settings.verbose,
    )

    # Extract results
    obj_space = res.F
    solutions = res.opt.get("solution", to_numpy=False)
    coeff_matrices = res.opt.get("coeff_matrix", to_numpy=False)
    alphas = res.opt.get("alphas", to_numpy=False)
    models = res.opt.get("models", to_numpy=False)
    theta_matrices = res.opt.get("theta", to_numpy=False)

    return IdentificationResult(
        problem=problem,
        result=res,
        objective_space=obj_space,
        solutions=solutions,
        coeff_matrices=coeff_matrices,
        alphas=alphas,
        models=models,
        theta_matrices=theta_matrices,
    )
