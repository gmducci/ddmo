from collections.abc import Callable
from itertools import permutations

import numpy as np

NamedFunc = tuple[Callable[..., float], str]

class FunctionLibrary:
    """
    Constructs a library of symbolic functions by applying a set of base functions 
    to all valid permutations of input variables.

    Attributes:
        base_func_generator (Callable[[List[float]], List[NamedFunc]]): A function that returns base functions.
        base_funcs (List[NamedFunc]): The actual list of base functions after calling base_func_generator.
        library (List[NamedFunc]): The expanded list of functions after applying permutations.
    """

    def __init__(self, base_func_generator: Callable[[list[float]], list[NamedFunc]]):
        self.base_func_generator = base_func_generator
        self.base_funcs: list[NamedFunc] = []
        self.library: list[NamedFunc] = []

    def set_base_funcs(self, parameters: tuple[float, float]):
        self.base_funcs = self.base_func_generator(list(parameters))

    def build(self, n_vars: int):
        self.library.clear()
        var_names = [f"x_{i}" for i in range(n_vars)]

        arity_to_permutations = {}
        for func, _ in self.base_funcs:
            arity = func.__code__.co_argcount
            if arity not in arity_to_permutations:
                arity_to_permutations[arity] = list(permutations(range(n_vars), arity))

        new_library = []
        for func, label in self.base_funcs:
            arity = func.__code__.co_argcount
            for idx_tuple in arity_to_permutations[arity]:
                wrapped_func = self._wrap_func_with_indices(func, idx_tuple)
                label_str = f"{label}({', '.join(var_names[i] for i in idx_tuple)})"
                new_library.append((wrapped_func, label_str))
        self.library = new_library

    @staticmethod
    def _wrap_func_with_indices(func: Callable, indices: tuple[int, ...]) -> Callable:
        def wrapped(*args):
            selected_args = tuple(args[i] for i in indices)
            return func(*selected_args)
        wrapped.__name__ = f"{func.__name__}_on_{'_'.join(map(str, indices))}"
        wrapped.__qualname__ = wrapped.__name__
        return wrapped

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected input of type np.ndarray, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D array, got array with shape {x.shape}")

        n_samples = x.shape[0]
        n_funcs = len(self.library)
        theta = np.empty((n_samples, n_funcs), dtype=float)

        for j, (func, _) in enumerate(self.library):
            theta[:, j] = np.fromiter((func(*row) for row in x), dtype=float, count=n_samples)

        return theta

    def get_rhs(self, state: np.ndarray, t: float, matrices: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Computes the right-hand side (RHS) of the differential system using the constructed function library 
        and a coefficient matrix.

        Args:
            state (np.ndarray): The state vector at time t, shape (n_vars,) or (n_samples, n_vars).
            t (float): Current time (unused, included for interface consistency).
            matrices (Tuple[np.ndarray, np.ndarray]): A tuple containing:
                - the input variables used to build the library (same shape as `state`)
                - the coefficient matrix, shape (n_features, n_outputs)

        Returns:
            np.ndarray: The evaluated RHS, shape (n_samples, n_outputs) or (n_outputs,) if state is 1D.
        """
        input_vars, coeff_matrix = matrices
        theta = self.evaluate(np.atleast_2d(state))
        rhs = theta @ coeff_matrix

        if state.ndim == 1:
            return rhs[0]
        return rhs




    def get_labels(self) -> list[str]:
        return [label for _, label in self.library]


# MAIN EXECUTION
if __name__ == "__main__":
    # Replace this import with your actual function source or definition
    from ddmo.library_functions import functions_template

    def get_base_functions(params: list[float]) -> list[NamedFunc]:
        alpha, beta = params
        return [
            (functions_template.identity, "x"),
            (lambda x, y: x**alpha * y**beta, f"x^{alpha} * y^{beta}")
        ]

    lib = FunctionLibrary(get_base_functions)
    lib.set_base_funcs((1.0, 2.0))
    lib.build(n_vars=2)

    x = np.ones((100, 2))
    x[:, 1] = 2
    theta = lib.evaluate(x)

    print("Theta shape:", theta.shape)
    print("Labels:", lib.get_labels())
