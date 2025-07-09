from dataclasses import dataclass, field
from typing import Any
from ddmo.solver.registry import SOLVER_REGISTRY
from ddmo.solver.base import BaseSolver

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ProblemSettings:
    """
    Configuration for the optimization problem to be solved.

    Attributes:
        n_var (int): Number of decision variables.
        n_obj (int): Number of objective functions.
        xl (Any): Lower bound(s) for decision variables. Can be scalar or array-like.
        xu (Any): Upper bound(s) for decision variables. Can be scalar or array-like.
        n_ieq_constr (int): Number of inequality constraints. Default is 0.
        n_eq_constr (int): Number of equality constraints. Default is 0.
        vtype (Optional[type]): Data type of decision variables (e.g., float, int). Default is float.
        elementwise (bool): Whether the evaluation is elementwise. Default is False.
        requires_kwargs (bool): Whether evaluation requires keyword arguments. Default is False.
        strict (bool): Enforce strict validation of input/output. Default is True.
    """
    n_var: int
    n_obj: int
    xl: Any = 0.0
    xu: Any = 1.0
    n_ieq_constr: int = 0
    n_eq_constr: int = 0
    vtype: Optional[type] = float
    elementwise: bool = False
    requires_kwargs: bool = False
    strict: bool = True


@dataclass
class AlgorithmSettings:
    """
    Configuration for the NSGA2 evolutionary algorithm.

    Attributes:
        pop_size (int): Size of the population.
        sampling (Any): Sampling strategy (e.g., FloatRandomSampling()).
        selection (Any): Selection operator (e.g., TournamentSelection()).
        crossover (Any): Crossover operator (e.g., SBX()).
        mutation (Any): Mutation operator (e.g., PM()).
        survival (Any): Survival strategy (e.g., RankAndCrowding()).
        output (Any): Output interface (e.g., MultiObjectiveOutput()).
        eliminate_duplicates (bool): Whether to remove duplicate individuals.
        advance_after_initial_infill (bool): Advance after the initial population is evaluated.
        tournament_type (Optional[str]): Type of tournament selection (used for diagnostics).
    """
    pop_size: int = 100
    sampling: Any = None
    selection: Any = None
    crossover: Any = None
    mutation: Any = None
    survival: Any = None
    output: Any = None
    eliminate_duplicates: bool = True
    advance_after_initial_infill: bool = True
    tournament_type: Optional[str] = None
    def __post_init__(self):
        assert callable(self.sampling), "sampling must be a callable sampling operator (e.g., FloatRandomSampling())"
        assert callable(self.crossover), "crossover must be a valid crossover operator"
        assert callable(self.mutation), "mutation must be a valid mutation operator"


@dataclass
class TerminationSettings:
    """
    Termination criteria for the optimization algorithm.

    Attributes:
        n_gen (int): Number of generations (iterations) until termination.
    """
    n_gen: int = 500


@dataclass
class GASettings:
    """
    Bundled configuration for running a multi-objective genetic algorithm (e.g., NSGA2).

    Attributes:
        problem (ProblemSettings): Problem-specific configuration.
        algorithm (AlgorithmSettings): Evolutionary algorithm configuration.
        termination (TerminationSettings): Stopping criteria.
        seed (Optional[int]): Random seed for reproducibility.
        verbose (bool): Whether to enable verbose output during optimization.
    """
    problem: ProblemSettings
    algorithm: AlgorithmSettings
    termination: TerminationSettings
    seed: Optional[int] = None
    verbose: bool = True