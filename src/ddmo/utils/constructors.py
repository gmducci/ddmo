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
    
    Separates library parameters (for function library) from formulation 
    parameters (for weak/strong formulation) for clarity and safety.

    Attributes:
        library_bounds: Bounds for library parameters as {name: (min, max)}.
            These parameters are passed to the library function as a dict.
        formulation_bounds: Bounds for formulation parameters as {name: (min, max)}.
            These control the weak formulation (stride, degree, window_length).
        n_obj: Number of objective functions.
        n_gen: Number of generations to run.
        algorithm: Algorithm name. Options: "NSGA2", "NSGA3". Default: "NSGA2".
        pop_size: Population size. Default: 100.
        crossover_eta: SBX crossover distribution index. Default: 15.0.
        crossover_prob: Crossover probability. Default: 0.9.
        mutation_eta: Polynomial mutation distribution index. Default: 20.0.
        seed: Random seed for reproducibility. Default: None.
        verbose: Whether to print progress. Default: True.

    Example:
        OptimizerSettings(
            library_bounds={"gamma_1": (8.4, 9.6), "gamma_2": (1.0, 10.0)},
            formulation_bounds={"stride": (5, 120), "degree": (3, 7)},
            n_obj=2,
            n_gen=100
        )
    """
    # Required parameters
    library_bounds: dict[str, tuple[float, float]]
    formulation_bounds: dict[str, tuple[float, float]]
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
        """Validate bounds format."""
        for name, bounds in {**self.library_bounds, **self.formulation_bounds}.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"Bounds for '{name}' must be a (min, max) tuple")
            if bounds[0] > bounds[1]:
                raise ValueError(f"Lower bound for '{name}' exceeds upper bound")
    
    @property
    def n_var(self) -> int:
        """Total number of decision variables."""
        return len(self.library_bounds) + len(self.formulation_bounds)
    
    @property
    def all_bounds(self) -> dict[str, tuple[float, float]]:
        """Combined bounds dict (library first, then formulation)."""
        return {**self.library_bounds, **self.formulation_bounds}
    
    @property
    def xl(self) -> dict[str, float]:
        """Lower bounds as a flat dict."""
        return {k: v[0] for k, v in self.all_bounds.items()}
    
    @property
    def xu(self) -> dict[str, float]:
        """Upper bounds as a flat dict."""
        return {k: v[1] for k, v in self.all_bounds.items()}
    
    @property
    def library_param_names(self) -> list[str]:
        """Names of library parameters."""
        return list(self.library_bounds.keys())
    
    @property
    def formulation_param_names(self) -> list[str]:
        """Names of formulation parameters."""
        return list(self.formulation_bounds.keys())


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