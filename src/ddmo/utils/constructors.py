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
    xl_dict: Any = 0.0
    xu_dict: Any = 1.0
    n_ieq_constr: int = 0
    n_eq_constr: int = 0
    vtype: Optional[type] = float
    elementwise: bool = False
    requires_kwargs: bool = False
    strict: bool = True


@dataclass
class AlgorithmSettings:
    """
    Configuration for the evolutionary optimization algorithm.

    Attributes:
        algorithm_name (str): Name of the algorithm to use. Options: "NSGA2", "NSGA3".
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
    algorithm_name: str = "NSGA2"
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


@dataclass
class OptimizerSettings:
    """
    Unified configuration for the optimization process.
    
    This is the recommended user-facing API that consolidates problem,
    algorithm, and termination settings with sensible defaults.

    Attributes:
        xl (dict[str, float]): Lower bounds for optimization variables.
        xu (dict[str, float]): Upper bounds for optimization variables.
        n_obj (int): Number of objective functions.
        n_gen (int): Number of generations to run.
        algorithm (str): Algorithm name. Options: "NSGA2", "NSGA3". Default: "NSGA2".
        pop_size (int): Population size. Default: 100.
        crossover_eta (float): SBX crossover distribution index. Default: 15.0.
        crossover_prob (float): Crossover probability. Default: 0.9.
        mutation_eta (float): Polynomial mutation distribution index. Default: 20.0.
        seed (int | None): Random seed for reproducibility. Default: None.
        verbose (bool): Whether to print progress. Default: True.
    """
    # Required parameters
    xl: dict[str, float]
    xu: dict[str, float]
    n_obj: int
    n_gen: int
    
    # Optional with sensible defaults
    algorithm: str = "NSGA2"
    pop_size: int = 100
    crossover_eta: float = 15.0
    crossover_prob: float = 0.9
    mutation_eta: float = 20.0
    seed: int | None = None
    verbose: bool = True
    
    def __post_init__(self):
        """Validate that bounds have matching keys."""
        if set(self.xl.keys()) != set(self.xu.keys()):
            raise ValueError("xl and xu must have the same keys")
        for key in self.xl:
            if self.xl[key] > self.xu[key]:
                raise ValueError(f"Lower bound for '{key}' exceeds upper bound")
    
    @property
    def n_var(self) -> int:
        """Number of decision variables, inferred from bounds."""
        return len(self.xl)


@dataclass
class IdentificationResult:
    """
    Structured result from the identify() function.
    
    Provides named access to all outputs from the sparse identification process,
    replacing the previous 8-element tuple return value.

    Attributes:
        problem: The optimization problem instance.
        result: The pymoo Result object containing full optimization details.
        objective_space: Array of objective function values for Pareto-optimal solutions.
        solutions: List of reconstructed trajectory solutions.
        coeff_matrices: List of coefficient matrices from sparse regression.
        alphas: List of optimized parameter values.
        models: List of fitted models (e.g., PySINDy models).
        theta_matrices: List of feature matrices used in regression.
    """
    problem: Any
    result: Any
    objective_space: Any  # np.ndarray
    solutions: list
    coeff_matrices: list
    alphas: list
    models: list
    theta_matrices: list