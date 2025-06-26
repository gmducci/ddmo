# compute_tajectory.py
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, RobustScaler, QuantileTransformer
import pysindy as ps
from scipy.integrate import odeint
from ..feature_library.weak_pde_library import WeakPDELibrary

def solver(Problem, optimization_vars, regression):
    """
    Compute the state space trajectory.

    The function performs the numerical integration of the right-end-side of the 
    reconstructed dynamical system, and outputs the resulting trajectory.

    Args:
        x_0 (list): List of initial conditions of the trajectory.
        time_array (float): Time array of the trajectory.
        *args: Description of additional positional arguments.
        **kwargs: Description of additional keyword arguments.

    Returns:
        trajectory (float): Description of the return value. Include details if it is
        complex, such as a dictionary with specific keys.

    Examples:
        Example usage of the function with inputs and expected outputs.

        >>> result = my_function(param1, param2)
        >>> print(result)
        expected_output
    """
    if len(optimization_vars) == 1:
        opt_vars, = optimization_vars  # Unpack if it's a 1D tuple
    else:
        opt_vars = optimization_vars  # Leave it as is otherwise

    Problem.alphas.append(opt_vars)

    if Problem.weak_form == False:

        Problem.library_object.set_base_funcs((opt_vars[0], opt_vars[1]))
        Problem.library_object.build(Problem.dimension)
        theta = Problem.library_object.evaluate(np.vstack(Problem.x_train))
        # theta = theta.T
        if Problem.scale_theta == True:
            scaler = StandardScaler()
            theta_scaled = scaler.fit_transform(theta)
            # Inverting the system using sparse regression techniques
            regression.fit(theta_scaled, Problem.x_dot)
            intercept = regression.intercept_
            coeff_matrix = regression.coef_.T  # [n x d] n = length library, d = dimension of the system

            coeff_matrix_scaled = coeff_matrix / scaler.scale_[:, None]  # Rescale back to original scale

            # Reconstruct the system
            regression_coeff = coeff_matrix_scaled
            sol_list = []
            for x_0_ in Problem.initial_values:
                t_array = Problem.time_array
                sol_ = odeint(Problem.library_object.get_rhs, x_0_, t_array, args=([opt_vars, regression_coeff],))

                sol_list.append(sol_)
            sol = np.vstack(sol_list)

            #self.coefficient_matrix.append(coeff_matrix_scaled)
        else:
            regression.fit(theta, Problem.x_dot)
            intercept = regression.intercept_
            coeff_matrix = regression.coef_.T  # [n x d] n = length library, d = dimension of the system
            regression_coeff = coeff_matrix
            # Reconstruct the system
            sol_list = []
            for x_0_ in Problem.initial_values:
                t_array = Problem.time_array
                sol_ = odeint(Problem.library_object.get_rhs, x_0_, t_array, args=([opt_vars, coeff_matrix],))

                sol_list.append(sol_)
            sol = np.vstack(sol_list)
    else:
        
        custom_lib = Problem.library_object.library(opt_vars)
        
        ode_lib = WeakPDELibrary(
            library_functions = Problem.library_object.library(opt_vars).functions,
            function_names = Problem.library_object.library(opt_vars).function_names,
            spatiotemporal_grid=Problem.time_array,
            is_uniform=True,
            K=5000,
            p=4)
        
        input_features = []
        input_indices = []

        for i in range(len(ode_lib.functions)):
            combinations = ode_lib._combinations(Problem.dimension,  # to be replaced with the problem dimension
                                                ode_lib.functions[i].__code__.co_argcount,
                                                True)
            a = list(combinations)
            input_indices_temp = [list(t) for t in a]
            number_expansion = len(a)
            input_indices.append(input_indices_temp)
            input_features.append([1] * number_expansion)

        input_features[0][0] = 0
        library_feature_input = [[indices, features] for indices, features in zip(input_indices, input_features)]

        ode_lib.library_feature_input = library_feature_input

        ode_lib.fit(Problem.x_train)


        n_targets = 1
        n_features = 2
        constr_rhs = np.asarray([0])
        constr_lhs = np.zeros((1, n_targets * n_features))

        optimizer = ps.SR3( threshold=.06,
                            thresholder="l1",
                            max_iter=20000,
                            normalize_columns=True,
                            tol=1e-4)

        input_features.clear()
        input_indices.clear()
        model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
        model.fit(Problem.x_train)
        regression_coeff = model.coefficients()
        regression_coeff = regression_coeff.T
        sol_list = []
        def get_rhs_local(ssp, t):
            ode_lib.fit(ssp)
            theta = ode_lib.transform_simple(ssp)
            rhs = theta @ regression_coeff
            return rhs

        for x_0_ in Problem.initial_values:

            sol_ = odeint(get_rhs_local, x_0_, Problem.time_array)

            sol_list.append(sol_)
        sol = np.vstack(sol_list)

    if Problem.target == 'trajectory':
        x_ref = Problem.x_raw
        x = sol

    elif Problem.target == 'derivative':
        x_ref = Problem.x_dot
        rhs = theta @ coeff_matrix
        x = rhs
    else:
        raise ValueError("Invalid target specified. Must be 'trajectory' or 'derivative'.")
    
    return x, x_ref, regression_coeff, opt_vars

