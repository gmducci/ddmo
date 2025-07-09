import numpy as np
import pymoo.gradient.toolbox as anp
import pysindy as ps
from pysindy.optimizers import SR3, STLSQ, ConstrainedSR3, EnsembleOptimizer
from sklearn.linear_model import Lasso
from ddmo.optimizer.optimizer import (
    Optimizer, 
    EvaluationResult)
from ddmo.utils.constructors import DataConstructor


class Pysindy_Solver:
    def __init__(self, 
                 data_instance: DataConstructor,
                 optimizer: str,
                 solver_parameters: dict):
        
        self.data_instance = data_instance
        self.solver_parameters = solver_parameters
        self.optimizer = optimizer

    def solve(self, optimization_vars):

        if len(optimization_vars) == 1:
            opt_vars, = optimization_vars  # Unpack if it's a 1D tuple
        else:
            opt_vars = optimization_vars  # Leave it as is otherwise

        # Optimizer dispatch logic
        _optimizers = {
            "STLSQ": STLSQ,
            "SR3": SR3,
            "ConstrainedSR3": ConstrainedSR3,
            "EnsembleOptimizer": EnsembleOptimizer,
            "Lasso": Lasso}
        
        if np.isscalar(opt_vars):
            opt_vars = (opt_vars,)
        # self.alphas.append(opt_vars)
        if np.isscalar(opt_vars) or isinstance(opt_vars, np.float64):
            opt_vars = (opt_vars,)

        named_funcs = self.data_instance.library_object([*opt_vars])
        functions = [f for f, _ in named_funcs]
        function_names = [name for _, name in named_funcs]

        lib_custom = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names
        )
        lib_custom.fit(self.data_instance.x_train)
        dt = self.data_instance.time_array[1] - self.data_instance.time_array[0]
        params_dict = self.solver_parameters
        optimizer = _optimizers[self.optimizer](**params_dict)

        # Increase threshold to enforce more sparsity
        try:

            model = ps.SINDy(feature_library=lib_custom, optimizer=optimizer)
            if isinstance(self.data_instance.x_train[0], (list, tuple, np.ndarray)):
                X = np.vstack(self.data_instance.x_train)
            else:
                X = self.data_instance.x_train
            model.fit(X, t=dt, x_dot=self.data_instance.x_dot)

            regression_coeff = model.coefficients()
            regression_coeff = regression_coeff.T
            sol_list = []

            for i in range(len(self.data_instance.x_0)):

                sol_ = model.simulate(self.data_instance.x_0[i], self.data_instance.time_array)

                sol_list.append(sol_)
            sol = np.vstack(sol_list)

            x = sol

            return x, regression_coeff, opt_vars, model
        except Exception as e:
            print(f"Error during solver execution: {e}")
            return None, None, None, None