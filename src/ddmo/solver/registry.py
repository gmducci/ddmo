from ddmo.solver.pysindy_solver import Pysindy_Solver
from ddmo.solver.scikit_solver import ScikitSolver

SOLVER_REGISTRY = {
    "pysindy": Pysindy_Solver,
    "scikit": ScikitSolver
}