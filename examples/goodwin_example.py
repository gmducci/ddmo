import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from scipy.integrate import solve_ivp

from ddmo import identify
from ddmo.equations.goodwin_eqs import GoodwinModel
from ddmo.formulation.formulation_settings import FormulationSettings
from ddmo.formulation.registry import FORMULATION_REGISTRY
from ddmo.solver.registry import SOLVER_REGISTRY
from ddmo.solver.settings import SolverSettings
from ddmo.utils.constructors import (
    AlgorithmSettings,
    GASettings,
    ProblemSettings,
    TerminationSettings,
)
from ddmo.utils.data_constructor import DataConstructor

warnings.filterwarnings("ignore")


def generate_goodwin_data(t_max=100, dt=0.05, noise_level=0.001, x0=None):
    model = GoodwinModel()

    if x0 is None:
        x0 = [0.1, 0.1, 0.1]

    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)

    def fun(t, y):
        return model.dynamics(y, t)

    sol = solve_ivp(
        fun, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12
    )

    x_clean = sol.y.T
    time_array = sol.t

    # Calculate x_dot_gt
    x_dot_gt = np.array([model.dynamics(y, t) for y, t in zip(x_clean, time_array)])

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, x_clean.shape)
        x_raw = x_clean + noise
    else:
        x_raw = x_clean.copy()

    # Assuming x_train is the clean data or the same as raw if we want to train on it
    # For now, let's keep x_train as clean data for reference, or user might want to train on noisy data.
    # The original file loaded x_train and x_train_noise (x_raw).
    x_train = x_clean

    return time_array, x_train, x_raw, x_dot_gt, np.array(x0)


if __name__ == "__main__":
    # Generate data using Goodwin equations
    time_array, x_train, x_raw, x_dot_gt, x_0_train = generate_goodwin_data(
        t_max=100, dt=0.025, noise_level=0.1
    )

    # Plot x_raw in phase space

    # User's request: "combine the goodwin_eqs.py and so to use those data as training data"

    # In original file: x_dot_hat = np.load(BENCHMARK_FOLDER / 'x_dot_train.npy')
    # We will use x_dot_gt as x_dot_hat for now, or we could estimate it.
    x_dot_hat = x_dot_gt

    NamedFunc = tuple[Callable[[float, float], float], str]

    def get_base_functions(params: list[float]) -> list[NamedFunc]:
        gamma_1, gamma_2 = params[0:2]
        return [
            (lambda x: x, lambda x: x),
            (lambda x: x**2, lambda x: f"{x}^2"),
            (lambda x, y: x * y, lambda x, y: f"{x} * {y}"),
            (lambda x: (1 / (1 + x**gamma_1)), lambda x: f"1 / 1 + {x}^{gamma_1}"),
            (lambda x: np.exp(-gamma_2 * x), lambda x: f"exp( {-gamma_2} {x} )"),
        ]

    x_0 = [x_0_train]

    # Create a DataConstructor instance
    training_data = DataConstructor(
        x_train=x_raw[::, :],
        time_array=time_array[::],
        x_dot=x_dot_hat[::, :],
        library_object=get_base_functions,
        x_0=x_0,
        x_raw=x_raw[::, :],
    )

    # Definition of opt boundaries
    # --- 1) library parameters (used by get_base_functions) --------------
    library_lower_bounds = {
        "gamma_1": 8.4,
        "gamma_2": 1.0,
    }

    library_upper_bounds = {
        "gamma_1": 9.6,
        "gamma_2": 10.0,
    }

    # --- 2) formulation / test-function parameters -----------------------
    formulation_lower_bounds = {"stride": 10}

    formulation_upper_bounds = {"stride": 50}

    library_param_names = list(library_lower_bounds.keys())
    formulation_param_names = list(formulation_lower_bounds.keys())

    # --- 3) combined bounds passed to ProblemSettings (backward compatible)
    lower_bounds = {**library_lower_bounds, **formulation_lower_bounds}
    upper_bounds = {**library_upper_bounds, **formulation_upper_bounds}
    problem_settings = ProblemSettings(
        n_var=3, n_obj=2, xl_dict=lower_bounds, xu_dict=upper_bounds
    )

    algorithm_settings = AlgorithmSettings(
        pop_size=50,
        sampling=FloatRandomSampling(),  # default
        crossover=SBX(eta=15, prob=0.9),  # default
        mutation=PM(eta=20),
    )  # default

    termination_settings = TerminationSettings(n_gen=15)

    ga_settings = GASettings(
        problem=problem_settings,
        algorithm=algorithm_settings,
        termination=termination_settings,
        seed=None,
        verbose=True,
    )

    solver_settings = SolverSettings(
        solver_name="pysindy",
        regularization="STLSQ",
        regularization_parameters={"alpha": 0.05, "threshold": 0.1},
    )

    solver_class = SOLVER_REGISTRY[solver_settings.solver_name]

    solver_settings = solver_class(
        data_instance=training_data,
        optimizer=solver_settings.regularization,
        solver_parameters=solver_settings.regularization_parameters,
    )

    formulation_settings = FormulationSettings(
        formulation_name="weak",
        formulation_attributes={
            "test_functions_class": "HannWindow",
            "window_length": 501,
            "stride": 50,
        },
    )

    # Load the formulation class from the registry (should be handled in the main)
    formulation_class = FORMULATION_REGISTRY[formulation_settings.formulation_name]

    # Not agnostic right now!
    weak_formulation = formulation_class(
        time_array=training_data.time_array,
        x_train=training_data.x_train,
        class_attributes=formulation_settings.formulation_attributes,
    )

    # Load the test-function family class from the registry

    identification = identify(
        data_instance=training_data,
        ga_settings=ga_settings,
        solver_instance=solver_settings,
        formulation_instance=weak_formulation,
    )

    (
        problem,
        res,
        obj_space,
        solutions,
        coeff_matrices,
        alphas,
        sindy_model,
        theta_matrices,
    ) = identification

    print(alphas)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x_raw[:, 0], x_raw[:, 1], x_raw[:, 2], alpha=0.5)
    ax.plot(solutions[0][:, 0], solutions[0][:, 1], solutions[0][:, 2])

    ax.set_title("Phase Space of x_raw")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
