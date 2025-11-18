from collections.abc import Callable
from pprint import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
from ddmo.utils.data_constructor import DataConstructor
from ddmo.utils.constructors import (
    ProblemSettings,
    AlgorithmSettings,
    TerminationSettings,
    GASettings)
from ddmo.solver.registry import SOLVER_REGISTRY
from ddmo.solver.settings import SolverSettings
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.integrate import solve_ivp

from ddmo import BENCHMARK_FOLDER, identify
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    x_0_train = np.load(BENCHMARK_FOLDER / 'x_0.npy')
    x_raw = np.load(BENCHMARK_FOLDER / 'x_train_noise.npy')
    x_train= np.load(BENCHMARK_FOLDER / 'x_train.npy')
    x_dot_hat = np.load(BENCHMARK_FOLDER / 'x_dot_train.npy')
    time_array = np.load(BENCHMARK_FOLDER / 'time.npy')
    x_dot_gt = np.load(BENCHMARK_FOLDER / 'x_dot_gt.npy')

    def goodwin(t, X, n=9.5, k1=1.0, k2=0.2, k3=1.0, k4=0.15, k5=1.0, k6=0.1):
        """
        Canonical 3-variable Goodwin oscillator:
            dx/dt = k1 * (1 / (1 + z^n)) - k2 * x
            dy/dt = k3 * x - k4 * y
            dz/dt = k5 * y - k6 * z
        """
        x, y, z = X
        hill = 1.0 / (1.0 + z**n)
        return np.array([
            k1 * hill - k2 * x,
            k3 * x - k4 * y,
            k5 * y - k6 * z
        ])

    def simulate_goodwin(tmax=150.0, dt=0.01, x0=(0, 1, 1.),
                        n=9.5, k1=1.0, k2=0.2, k3=1.0, k4=0.15, k5=1.0, k6=0.1,
                        noise=0.02, seed=0):
        t = np.arange(0.0, tmax + dt, dt)

        def f(t, X): return goodwin(t, X, n, k1, k2, k3, k4, k5, k6)

        sol = solve_ivp(f, (t[0], t[-1]), np.array(x0, dtype=float),
                        t_eval=t, rtol=1e-9, atol=1e-12)
        X = sol.y.T
        if noise > 0:
            rng = np.random.default_rng(seed)
            scale = np.abs(X)  # magnitude of the signal
            X = X + noise  * rng.standard_normal(X.shape)
            #X[:, 0] = X[:, 0] + noise * rng.standard_normal(X.shape[0])
        return t, X

    # t, x_train = simulate_goodwin()

    NamedFunc = tuple[Callable[[float, float], float], str]
    def get_base_functions(params: list[float]) -> list[NamedFunc]:
        alpha, beta = params[0:2]
        return [
            (lambda x: x, lambda x: x),
            (lambda x: x**2, lambda x: f"{x}^2"),
            (lambda x: (1 / (1 + x**alpha)), lambda x: f"1 / 1 + {x}^{alpha}"),
            (lambda x: np.exp(- beta * x), lambda x: f"exp( {-beta} {x} )")
        ]

    x_0 = [x_0_train]

    # Create a DataConstructor instance
    training_data = DataConstructor(
        x_train=x_raw[:, :],
        time_array=time_array,
        x_dot=x_dot_hat[:, :],
        library_object=get_base_functions,
        x_0=x_0,
        x_raw=x_raw)
    
    # Define all the optimization settings in a constructor
    problem_settings = ProblemSettings(n_var=3,
                                       n_obj=2, 
                                       xl=np.array([8.4, 1.0, 10]),
                                       xu=np.array([9.6, 10.0, 200]))
    
    algorithm_settings = AlgorithmSettings(pop_size=60,
                                           sampling=FloatRandomSampling(),   # default
                                           crossover=SBX(eta=15, prob=0.9),  # default
                                           mutation=PM(eta=20))              # default
    
    termination_settings = TerminationSettings(n_gen=10)

    ga_settings = GASettings(problem=problem_settings,
                             algorithm=algorithm_settings,
                             termination=termination_settings, 
                             seed=None,
                             verbose=True)


    solver_settings = SolverSettings(solver_name="pysindy",
                                    optimizer="STLSQ",
                                    solver_parameters={
                                    "alpha": 0.05,
                                    "threshold": 0.1
                                    })

    solver_class = SOLVER_REGISTRY[solver_settings.solver_name]

    solver_settings = solver_class(data_instance=training_data,
                          optimizer=solver_settings.optimizer,
                          solver_parameters=solver_settings.solver_parameters)

    identification = identify(data_instance=training_data,
                              ga_settings=ga_settings,
                              solver_instance=solver_settings,                              
                              formulation="weak",
                              formulation_options={
                                # "function_class": None,
                                "time_array": time_array,
                                "x_train": x_train[:, :],
                                "window_length": 151,
                                "stride": 75
                              },
                              )

    (problem,
        res,
        obj_space,
        solutions,
        coeff_matrices,
        alphas,
        sindy_model,
        theta_matrices,
    ) = identification

    print(alphas)
