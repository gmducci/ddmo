import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from ddmo.optimizer.optimizer import Optimizer
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import pysindy as ps
import inspect

class DataGenerator:
    """
    DataGenerator for synthetic trajectories from ODE models with configurable noise and plotting.

    Parameters
    ----------
    model_cls : class
        A class providing `dynamics(x, t)` for ODE integration.
    param_ranges : dict
        Parameter ranges for model initialization, e.g. {'alpha': (1,3), 'beta': (1,3)}.
    state_dim : int, optional
        Dimension of the system state. If None, the model must define `state_dim` attribute.
    batch_size : int
        Number of trajectories to generate per call.
    time_array : np.ndarray, optional
        Time points for ODE integration. Defaults to np.linspace(0,1,50).
    noise_std : float
        Standard deviation of additive Gaussian noise.
    random_state : int or None
        Seed for reproducibility.
    """
    def __init__(
        self,
        model_cls,
        param_ranges,
        state_dim=None,
        batch_size=5,
        time_array=None,
        noise_std=0.2,
        random_state=None,
    ):
        self.model_cls = model_cls
        self.param_ranges = param_ranges
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.time_array = time_array if time_array is not None else np.linspace(0, 1, 50)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_state)

    def _sample_parameters(self):
        params = {}
        for name, (low, high) in self.param_ranges.items():
            params[name] = self.rng.uniform(low, high)
        return params

    def generate(self):
        """
        Generate noisy state trajectories and corresponding derivatives.

        Returns
        -------
        x_noisy_list : list of np.ndarray
            List of shape (batch_size, len(time_array), state_dim) noisy trajectories.
        x_dot : np.ndarray
            Array of shape (batch_size * len(time_array), state_dim) of true derivatives.
        time_array : np.ndarray
            Time points used for integration.
        initial_conditions : list of np.ndarray
            Initial states used for each trajectory.
        params : dict
            Ground-truth parameters sampled.
        """
        # Sample parameters and instantiate model
        params = self._sample_parameters()
        params = {'alpha': 2, 'beta': 3}
        model = self.model_cls(**params)

        # Determine state dimension
        if self.state_dim is not None:
            state_dim = self.state_dim
        elif hasattr(model, 'state_dim'):
            state_dim = model.state_dim
        else:
            raise AttributeError(
                "Cannot determine state dimension: please provide `state_dim` when initializing DataGenerator or implement `state_dim` in the model.`"
            )

        x_dot = np.zeros((self.batch_size * len(self.time_array), state_dim))
        x_noisy_list = []
        initial_conditions = []
        # Generate trajectories
        ic = [[1, 1, 0.05], [0.6, 0.8, 0.05], [0.8, 0.6, 0.05]]
        for i in range(self.batch_size):
            # Random initial state in [0,1]^state_dim
            x0 = self.rng.uniform(0, 2, size=state_dim)
            x0[-1] = 0.05  # Ensure last state variable is zero
            initial_conditions.append(ic[i])

            # Integrate ODE
            sol = odeint(model.dynamics, ic[i], self.time_array)

            # Add noise
            noise = self.rng.normal(scale=self.noise_std, size=sol.shape)
            sol_noisy = sol + noise
            x_noisy_list.append(sol_noisy)

            # Compute derivatives on noisy states
            for j, t in enumerate(self.time_array):
                x_dot[i * len(self.time_array) + j, :] = model.dynamics(sol_noisy[j], t)

        return x_noisy_list, x_dot, self.time_array, initial_conditions, params

    def plot_trajectories(self, x_noisy_list=None, figsize=(8,6)):
        """
        Plot noisy trajectories for each state variable.

        Parameters
        ----------
        x_noisy_list : list of np.ndarray, optional
            Trajectories to plot. If None, will generate new data.
        figsize : tuple
            Figure size.
        """
        if x_noisy_list is None:
            x_noisy_list, _, time_array, _, _ = self.generate()
        else:
            time_array = self.time_array

        n_states = x_noisy_list[0].shape[1]
        fig, axes = plt.subplots(n_states, 1, sharex=True, figsize=figsize)
        if n_states == 1:
            axes = [axes]

        for idx, traj in enumerate(x_noisy_list):
            for k in range(n_states):
                axes[k].plot(time_array, traj[:, k], label=f"Trajectory {idx}")
                axes[k].set_ylabel(f"State {k}")

        axes[-1].set_xlabel("Time")
        axes[0].legend()
        fig.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    from ddmo.equations.plugflow_equation import PlugFlow
    from ddmo.library_functions.funcs_refactored import FunctionLibrary

    # Define parameter ranges and known state dimension
    param_ranges = {'alpha': (1, 3), 'beta': (1, 3)}
    state_dim = 3

    # Create generator instance
    generator = DataGenerator(
        model_cls=PlugFlow,
        param_ranges=param_ranges,
        state_dim=state_dim,
        batch_size=3,
        time_array=np.linspace(0, 2, 150),
        noise_std=0.0,
        random_state=60
    )

    # Generate data
    x_noisy, x_dot, t, x0_list, params = generator.generate()
    print("Sampled parameters:", params)
    print("Initial conditions:", x0_list)
    X = np.vstack(x_noisy)
    dX = x_dot
    # Plot trajectories
    #generator.plot_trajectories(x_noisy)

    from ddmo import identify
    from ddmo.library_functions import functions_template 
    from typing import Callable, List, Tuple


    NamedFunc = Tuple[Callable[[float, float], float], str]
    def get_base_functions(params: List[float]) -> List[NamedFunc]:
        alpha, beta = params
        return [
            (lambda x: x, "x"),
            (lambda x: np.sin(x), "sin(x)"),
            (lambda x, y: x**alpha * y**beta, f"x^{alpha} * y^{beta}")
        ]
    
    named_funcs = get_base_functions([2, 3])
    functions = [f for f, _ in named_funcs]
    function_names = [lambda *args, name=name: name for _, name in named_funcs]

    lib_custom = ps.CustomLibrary(
        library_functions=functions,
        function_names=function_names
    )
    lib_custom.fit(x_noisy)

    model = ps.SINDy(feature_library=lib_custom)
    model.fit(X, t=t[1] - t[0], x_dot=dX)
    print("\nRecovered model:")
    model.print()
    # problem, res, obj_space, solutions, coeff_matrices, alphas = identify(
    #     x_train = x_noisy,
    #     x_raw=x_noisy,
    #     x_dot=x_dot,
    #     t = t,
    #     x0=x0_list,
    #     solver_parameters= {'alpha': 0.001},
    #     library_object=lib,
    #     bounds= [np.array([1, 1]), np.array([4, 4])],
    #     weak_form=False,
    #     scale_theta=True,
    #     solver='lasso',
    #     number_generations=20,
    #     population_size=25)