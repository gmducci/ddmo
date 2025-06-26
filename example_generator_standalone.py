import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from ddmo.optimizer.optimizer import SparseIdentification
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

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
        random_state=42
    )

    # Generate data
    x_noisy, x_dot, t, x0_list, params = generator.generate()
    print("Sampled parameters:", params)
    print("Initial conditions:", x0_list)

    # Plot trajectories
    #generator.plot_trajectories(x_noisy)

    from ddmo import identify
    from ddmo.library_functions import functions_template 
    from typing import Callable, List, Tuple


    NamedFunc = Tuple[Callable[[float, float], float], str]
    def get_base_functions(params: List[float]) -> List[NamedFunc]:
        alpha, beta = params
        return [
            (functions_template.identity, "x"),
            (lambda x, y: x**alpha * y**beta, f"x^{alpha} * y^{beta}"),
        ]

    library_object = FunctionLibrary(get_base_functions)

    problem = SparseIdentification(x_noisy,
                                t,
                                x_dot,
                                library_object,
                                x0_list,
                                x_noisy,
                                regression_model='lasso',
                                alpha_lasso = 1e-4,
                                #alpha_lasso=0.000005,
                                target='trajectory',
                                scale_theta= False,
                                # solver = pysindy, pysindy_weak, scikit
                                weak_form=False, # to remove
                                objectives=None)
    

    problem.n_obj = 2
    problem.n_var = 2
    problem.dimension = 3

    problem.xl = np.array([1, 1])
    problem.xu = np.array([4, 4])

    
    algorithm = NSGA2(pop_size=25)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 15))
    
    obj_space = res.F
    solutions = res.opt.get("solution")
    coeff_matrices = res.opt.get("coeff_matrix")
    alphas = res.opt.get("alphas")


    # problem, res = identify(
    #     x=x_noisy,
    #     x_dot=x_dot,
    #     t=t,
    #     n_gen=10,
    #     x0=x0_list,
    #     regression_model="lasso",
    #     alpha_lasso=0.00005,
    #     library_funcs=library_object,
    #     weak_form=False
    # )
    '''
    problem, res = identify(
        x=x_noisy,
        t=t,
        x0=x0_list,
        bounds=((1.0,4.5),(1.0,4.5)),
        pop_size=30,
        n_gen=100,
        regression_model="ridge",
        alpha_lasso=0.05,
        scale_theta=False,
        target="derivative",
        library_families=["polynomial","trigonometric"],
        individual_funcs=[np.sin, np.cos],
        seed=123,
        verbose=True
    )
    '''

# -----------------------------------------------------------------------------
# Predefined function families
# -----------------------------------------------------------------------------
'''
PREDEFINED_FAMILIES = {
    'polynomial': [lambda x: x, lambda x: x**2, lambda x: x**3],
    'trigonometric': [np.sin, np.cos],
    'exponential': [np.exp],
    'custom': None,
    'linear': [lambda x: x],
    'quadratic': [lambda x: x**2],
    'cubic': [lambda x: x**3],
    'quartic': [lambda x: x**4],
    'quintic': [lambda x: x**5],
    'reciprocal': [lambda x: 1/x],
    'reciprocal_sqrt': [lambda x: 1/np.sqrt(x)],
    'sqrt': [np.sqrt],
    'cbrt': [np.cbrt],
    'logarithmic': [np.log],
    'sigmoid': [lambda x: 1/(1+np.exp(-x))],
    'gaussian': [lambda x: np.exp(-x**2)],
    'hyperbolic': [np.sinh, np.cosh, np.tanh],
    'softplus': [lambda x: np.log1p(np.exp(x))],
    'relu': [lambda x: np.maximum(0, x)],
    'absolute': [np.abs],
    'polynomial_harmonic': [lambda x: x * np.sin(x), lambda x: x * np.cos(x)],
    'piecewise_linear': [lambda x: np.where(x >= 0, x, 0.5 * x)],
    'rational': [lambda x: (x**2 + 1) / (x + 1)],
    'erf': [erf]
}
'''

