# functions.py
import numpy
import numpy as np
import pywt
import pynumdiff
import pynumdiff.optimize
from scipy.optimize import minimize
from typing import List, Tuple, Callable


NamedFunc = Tuple[Callable[[float, float], float], str]

class LibraryFunctions:
    def __init__(self, function_generator: Callable[[float, float], List[NamedFunc]]):
        """
        Parameters
        ----------
        function_generator : Callable
            A function like poly_family(max_degree), which returns a function f(alpha, beta)
        """
        self.function_generator = function_generator
        self.basis_functions: List[NamedFunc] = []

    def generate_functions(self, alpha: float, beta: float):
        """
        Stores basis functions internally based on alpha and beta
        """
        self.basis_functions = self.function_generator(alpha, beta)

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate stored basis functions on given data.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, 2)

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        n_samples = data.shape[0]
        n_features = len(self.basis_functions)
        X = np.empty((n_samples, n_features))
        for j, (func, _) in enumerate(self.basis_functions):
            X[:, j] = np.array([func(x, y) for x, y in data])
        return X

    def get_labels(self) -> List[str]:
        return [label for _, label in self.basis_functions]


class LibraryFunctions_bak:

    def __init__(self, function_list = None):
        self.function_list = function_list
    def _functions(self, alpha, beta = None, gamma = None):
        return self.function_list(alpha)
    
    def evaluate_poly_family(
        data: np.ndarray,
        poly_funcs: List[Tuple[callable, str]]
    ) -> np.ndarray:
        """
        Evaluates the functions from poly_family on a dataset.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, 2)
            Each row corresponds to a point (x, y).
        
        poly_funcs : List of (function, label)
            Output from poly_family(...)(alpha, beta).

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Matrix where each column is the evaluation of one basis function.
        """
        X = np.empty((data.shape[0], len(poly_funcs)))
        for j, (func, _) in enumerate(poly_funcs):
            X[:, j] = np.array([func(x, y) for x, y in data])
        return X

    def build_theta(self, train_data, alpha):
        library = self._functions(alpha)[0]

        lib_entries = len(library)

        theta = np.zeros([lib_entries, len(train_data)])
        for i in range(len(train_data)):
            for j in range(len(library)):
                arg_names = library[j].__code__.co_varnames[:library[j].__code__.co_argcount]
                arg_indices = [int(arg.split('_')[-1]) if '_' in arg else 0 for arg in arg_names]  # Extract indices from argument names
                if len(arg_indices) == 1:
                    theta[j, i] = library[j](train_data[i, arg_indices[0]])
                if len(arg_indices) == 2:
                    theta[j, i] = library[j](train_data[i, arg_indices[0]], train_data[i, arg_indices[1]])

        return theta

    def get_theta_raw(self, x, alpha):
        library = self._functions(alpha)[0]

        lib_entries = len(library)

        theta_raw = np.zeros(lib_entries)
        for j in range(len(library)):
            arg_names = library[j].__code__.co_varnames[:library[j].__code__.co_argcount]
            arg_indices = [int(arg.split('_')[-1]) if '_' in arg else 0 for arg in arg_names]  # Extract indices from argument names
            if len(arg_indices) == 1:
                theta_raw[j] = library[j](x[arg_indices[0]])
            if len(arg_indices) == 2:
                theta_raw[j] = library[j](x[arg_indices[0]], x[arg_indices[1]])

        return theta_raw

    def get_rhs(self, ssp, t, matrices):
        lib = self._functions(matrices[0])[0]
        coeff_matrix = matrices[1]
        # No need to recompute theta!!! Take raw-wise elements from build_theta output!!!
        theta = self.get_theta_raw(ssp, matrices[0])
        #theta = [lib[0](x), lib[1](y), lib[2](z), lib[3](z), lib[4](x, y), lib[5](x), lib[6](x, z)]
        rhs = theta @ coeff_matrix
        
        return rhs
    
class LibrarySindy:

    def __init__(self, library, function_list = None):
        self.library = library
        self.function_list = function_list

