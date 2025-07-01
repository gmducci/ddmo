import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pynumdiff
import pynumdiff.optimize
import pywt
import scipy.stats as stats
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse import linalg as splin

_has_matplotlib = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    _has_matplotlib = False
    logging.warning("Matplotlib is not installed - plotting "
                    "functionality disabled")

class Derivatives:

    def __init__(self, method, time_array, optimization = True, cutoff = 3):
        self.method = method
        self.time_array = time_array
        self.dt = time_array[1] - time_array[0]
        self.optimization = optimization
        self.cutoff = cutoff

    def _get_tvgamma(self):
        dt = self.time_array[1] - self.time_array[0]
        cutoff_frequency = self.cutoff  # high frequency of signal in the data.
        # estimate by (a) counting real # peaks per second in the data or (b) look at power spectra and choose cutoff.
        log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
        tvgamma = np.exp(log_gamma)
        return tvgamma

    def optimize_column(self, i, x_col):
        params, val = pynumdiff.optimize.kalman_smooth.constant_velocity(x_col, self.dt, params=None,
                                                                         tvgamma=self._get_tvgamma())
        print(f"Optimized Params for column {i}: {params}")
        return params

    def process_column(self, x_col, params):
        x_hat_col, dxdt_hat_col = pynumdiff.kalman_smooth.constant_velocity(x_col, self.dt, params)
        return x_hat_col, dxdt_hat_col

    def run(self, x):
        self.x = x
        x_hat = np.zeros_like(x)
        dxdt_hat = np.zeros_like(x)

        # Step 1: Perform optimization in serial
        params_list = [[1e-8, 0.1]] * x.shape[1]  # Default params for non-optimized casedef optimize_column_wrapper(args):

        def optimize_column_wrapper(args):
            return self.optimize_column(*args)

        if self.method == "kalman" and self.optimization:
            with ThreadPoolExecutor(max_workers=10) as executor:
                params_list = list(executor.map(optimize_column_wrapper, [(i, x[:, i]) for i in range(x.shape[1])]))

        # Step 2: Perform filtering in parallel
        with Pool(processes=10) as pool:
            results = pool.starmap(self.process_column, [(x[:, i], params_list[i]) for i in range(x.shape[1])])

        for i, (x_hat_col, dxdt_hat_col) in enumerate(results):
            x_hat[:, i] = x_hat_col
            dxdt_hat[:, i] = dxdt_hat_col

        return x_hat, dxdt_hat
    def get_derivatives(self, x):
        self.optimization= False
        x_hat = np.zeros_like(x)
        dxdt_hat = np.zeros_like(x)

        if self.method == "kalman":

            if self.optimization == True:
                for i in range(len(x[0, :])):
                    params, val = pynumdiff.optimize.kalman_smooth.constant_acceleration(x[:, i], self.dt, params=None,
                                                                                     tvgamma=self._get_tvgamma(),  # hyperparameter
                                                                                     dxdt_truth=None)
                                                                                     # no ground truth data
                                                                                     # options={})

                    print(params)

                    x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_acceleration(x[:, i], self.dt, params)
            if self.optimization == False:

                for i in range(len(x[0, :])):
                    x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_acceleration(x[:, i], self.dt, [1e-8, 1e-8])

            return x_hat, dxdt_hat

        elif self.method == "polydiff":
            if self.optimization == True:
                for i in range(len(x[0, :])):
                    params, val = pynumdiff.optimize.linear_model.polydiff(x[:, i], self.dt, params=None,
                                                                           tvgamma=self._get_tvgamma(),
                                                                           dxdt_truth=None)
                    print('Optimal parameters: ', params)
                    x_hat[:, i], dxdt_hat[:, i] = pynumdiff.linear_model.polydiff(x[:, i], self.dt, params)
            if self.optimization == False:

                for i in range(len(x[0, :])):
                    x_hat[:, i], dxdt_hat[:, i] = pynumdiff.smooth_finite_difference.splinediff(x[:, i], self.dt, [1e-8, 1e-8])

            return x_hat, dxdt_hat


        elif self.method == "FD-1st":
            for i in range(len(x[0, :])):

                params = [50]
                x_hat[:, i], dxdt_hat[:, i] = pynumdiff.finite_difference.first_order(x[:, i], self.dt, params, options={'iterate': False})
            return x_hat, dxdt_hat


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _estimate_tvgamma(dt: float, cutoff_frequency: float = 3.0) -> float:
    """Return a heuristic *tvgamma* value for *pynumdiff* Kalman smoothers."""
    log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
    return float(np.exp(log_gamma))


def add_relative_noise(x: np.ndarray, noise_level: float = 0.05, *, rng: np.random.Generator | None = None) -> np.ndarray:  # noqa: D401,E501
    """Return *x* with i.i.d Gaussian noise proportional to its terminal variance.

    Parameters
    ----------
    x
        Array of shape ``(n_samples, n_features)``.
    noise_level
        Fraction of the terminal‑segment standard deviation to use as the noise
        scale (e.g. ``0.05`` means *5 %* of the variance).
    rng
        Optional :pyclass:`numpy.random.Generator` for reproducibility.
    """
    rng = np.random.default_rng() if rng is None else rng
    noisy = np.empty_like(x)
    tail = x[int(0.7 * len(x)) :]  # last 30 % of the trace

    for j in range(x.shape[1]):
        sigma = tail[:, j].std()
        noisy[:, j] = x[:, j] + rng.normal(scale=noise_level * sigma, size=len(x))

    return noisy


def add_absolute_noise(x: np.ndarray, sigma: float = 1e-3, *, rng: np.random.Generator | None = None) -> np.ndarray:  # noqa: D401,E501
    """Return *x* with additive N(0, *sigma²*) white noise."""
    rng = np.random.default_rng() if rng is None else rng
    return x + rng.normal(scale=sigma, size=x.shape)




def add_noise(x_train, time_array, noise_level):
    x_train_noise = np.zeros_like(x_train)
    asymptotic_dynamics = round(len(x_train) * (3/3))
    for i in range(len(x_train[0, :])):
        rvs = stats.norm.rvs(loc=0, scale = x_train[-asymptotic_dynamics:, i].std(), size = time_array.size)
        x_train_noise[:, i] = x_train[:, i] + (rvs * noise_level)
        # x_train_noise[:, i] = x_train[:, i] + \
        #                       np.random.normal(0, x_train[-asymptotic_dynamics:, i].std(), time_array.size) * noise_level

    return x_train_noise

class WaveletDerivative:

    def __init__(self, x_train_noise, time_array, x_dot_true = None):
        """Initialize the class with attributes.

        Args:
            x_train_noise (float): Training data with noise.
            time_array (float): time array related to the training data.
            x_dot_true (float): Analytical computation of the derivatives (only available for synth. data)
        """

        self.x_train_noise = x_train_noise
        self.time_array = time_array
        self.x_dot_true = x_dot_true

    def pad_signal(self):
        """Pad the training data signal, in order to further perform wavelet transform.

        Args:

        Returns:
            float: padded signal.
        """

        padded_signal = np.zeros((len(self.x_train_noise)*2, len(self.x_train_noise[0, :])))
        for i in range(len(self.x_train_noise[0, :])):
            padded_signal[:, i] = np.pad(self.x_train_noise[:, i], (len(self.x_train_noise[:, 0]) // 2, len(self.x_train_noise[:, 0]) // 2),
                                   mode='symmetric', reflect_type='odd')
        return padded_signal

    def _obj_function(self, scale):
        """Internal method. Objective function for wavelet scales optimization
        It requires to know the analytical expression of the Equations. Only available for synth. data

        Args:
            scale (float): optimization variable
        Returns:
            float: Least squares error.
        """

        x_dot_wt = np.zeros_like(self.x_train_noise)

        # Compute the continuous wavelet transform with the given scales
        padded_signal = self.pad_signal()
        for i in range(len(self.x_train_noise[0, :])):
            dx_dt, _ = pywt.cwt(padded_signal[:, i], scales=scale, wavelet='gaus1', sampling_period=0.01)
            transformed_signal = -dx_dt.ravel()
            x_dot_wt[:, i] = transformed_signal[
                             len(self.x_train_noise[:, 0]) // 2:len(self.x_train_noise[:, 0]) // 2 + len(self.x_train_noise[:, 0])]

        # Calculate the difference between the transformed signal and the reference signal
        difference = x_dot_wt - self.x_dot_true

        return np.sum(difference ** 2)

    def optimize_scale(self):

        initial_value = 1
        optimization = minimize(self._obj_function, initial_value, method = 'Nelder-Mead')

        return optimization.x

    def get_wt_derivative(self):
        """Computes the numerical derivatives of the noisy signal using continuous wavelet method.

        Args:
        Returns:
            float: First order derivative of the noisy signal with continuous wavelet method.
        """

        x_dot_wt = np.zeros_like(self.x_train_noise)
        time_final = self.time_array[-1]
        # Pad the original time series to get rid of border effects
        padded_signal = self.pad_signal()
        scale = self.optimize_scale()

        for i in range(len(self.x_train_noise[0, :])):
            # The scale of the wavelet changes in the three dimensions, implement that!!!!!!!
            dx_dt, _ = pywt.cwt(padded_signal[:, i], scales=scale[0], wavelet='gaus1', sampling_period=0.01)
            # bi-orthogonal wt
            transformed_signal = -dx_dt.ravel()
            x_dot_wt[:, i] = transformed_signal[
                             len(self.x_train_noise[:, 0]) // 2:len(self.x_train_noise[:, 0]) // 2 + len(self.x_train_noise[:, 0])]

        print("scale is: ", scale[0])

        return x_dot_wt

class TVReg:
    def __init__(self, x_train_noise, time_array, x_dot_true = None):
        """Initialize the class with attributes.

        Args:
            x_train_noise (float): Training data with noise.
            time_array (float): time array related to the training data.
            x_dot_true (float): Analytical computation of the derivatives (only available for synth. data)
        """

        self.x_train_noise = x_train_noise
        self.time_array = time_array
        self.x_dot_true = x_dot_true

    def log_iteration(self, ii, s0, u, g):

        relative_change = np.linalg.norm(s0) / np.linalg.norm(u)
        g_norm = np.linalg.norm(g)
        logging.info(f'iteration {ii:4d}: relative change = {relative_change:.3e}, '
                     f'gradient norm = {g_norm:.3e}\n')

    def TVRegDiff(self, data, itern, alph, u0=None, scale='small', ep=1e-6, dx=None,
                  plotflag=False, diagflag=False, precondflag=True,
                  diffkernel='abs', cgtol=1e-4, cgmaxit=100):
        """
        Estimate derivatives from noisy data based using the Total
        Variation Regularized Numerical Differentiation (TVDiff)
        algorithm.

        Parameters
        ----------


        Returns
        -------
        u : ndarray
            Estimate of the regularized derivative of data.  Due to
            different grid assumptions, length(u) = length(data) + 1
            if scale = 'small', otherwise length(u) = length(data).
        """

        # Make sure we have a column vector

        data = np.array(data)
        assert len(data.shape) == 1, "data is not one-dimensional"
        # Get the data size.
        n = len(data)
        dx = self.time_array[1] - self.time_array[0]
        # Default checking. (u0 is done separately within each method.)
        if dx is None:
            dx = 1.0 / n

        # Different methods for small- and large-scale problems.
        if (scale.lower() == 'small'):

            # Differentiation operator
            d0 = -np.ones(n) / dx
            du = np.ones(n - 1) / dx
            dl = np.zeros(n - 1)
            dl[-1] = d0[-1]
            d0[-1] *= -1

            D = sparse.diags([dl, d0, du], [-1, 0, 1])
            DT = D.transpose()

            # Antidifferentiation and its adjoint
            def A(x):
                return (np.cumsum(x) - 0.5 * (x + x[0])) * dx

            def AT(x):
                return np.concatenate([[sum(x[1:]) / 2.0],
                                       (sum(x) - np.cumsum(x) + 0.5 * x)[1:]]) * dx

            # Default initialization is naive derivative

            if u0 is None:
                u0 = D * data

            u = u0.copy()
            # Since Au( 0 ) = 0, we need to adjust.
            ofst = data[0]
            # Precompute.
            ATb = AT(ofst - data)  # input: size n

            smoothed_signal = data.copy()  # Initialize smoothed signal

            # Main loop.
            for ii in range(1, itern + 1):
                if diffkernel == 'abs':
                    # Diagonal matrix of weights, for linearizing E-L equation.
                    Q = sparse.spdiags(1. / (np.sqrt((D * u) ** 2 + ep)), 0, n, n)
                    # Linearized diffusion matrix, also approximation of Hessian.
                    L = dx * DT * Q * D
                elif diffkernel == 'sq':
                    L = dx * DT * D
                else:
                    raise ValueError('Invalid diffkernel value')

                # Gradient of functional.
                print(alph)
                g = AT(A(u)) + ATb + alph * L * u

                # Prepare to solve linear equation.
                if precondflag:
                    # Simple preconditioner.
                    P = alph * sparse.spdiags(L.diagonal() + 1, 0, n, n)
                else:
                    P = None

                def linop(v):
                    return (alph * L * v + AT(A(v)))

                linop = splin.LinearOperator((n, n), linop)

                s, info_i = sparse.linalg.cg(
                    linop, g, x0=None, tol=cgtol, maxiter=cgmaxit,
                    callback=None, M=P, atol='legacy')

                if diagflag:
                    self.log_iteration(ii, s[0], u, g)
                    if (info_i > 0):
                        logging.warning(
                            "WARNING - convergence to tolerance not achieved!")
                    elif (info_i < 0):
                        logging.warning("WARNING - illegal input or breakdown")

                # Update solution.
                u = u - s
                smoothed_signal = data[0] + (np.cumsum(u) * dx)
                # offset = np.mean(data) - np.mean(smoothed_signal)
                # smoothed_signal = smoothed_signal + offset

                # Display plot.
                if plotflag:
                    plt.plot(u)
                    plt.show()

        elif (scale.lower() == 'large'):

            # Construct anti-differentiation operator and its adjoint.
            def A(v):
                return np.cumsum(v)

            def AT(w):
                return (sum(w) * np.ones(len(w)) -
                        np.transpose(np.concatenate(([0.0],
                                                     np.cumsum(w[:-1])))))

            # Construct differentiation matrix.
            c = np.ones(n)
            D = sparse.spdiags([-c, c], [0, 1], n, n) / dx
            mask = np.ones((n, n))
            mask[-1, -1] = 0.0
            D = sparse.dia_matrix(D.multiply(mask))
            DT = D.transpose()
            # Since Au( 0 ) = 0, we need to adjust.
            data = data - data[0]
            # Default initialization is naive derivative.
            if u0 is None:
                u0 = np.concatenate(([0], np.diff(data)))
            u = u0
            # Precompute.
            ATd = AT(data)

            # Main loop.
            for ii in range(1, itern + 1):

                if diffkernel == 'abs':
                    # Diagonal matrix of weights, for linearizing E-L equation.
                    Q = sparse.spdiags(1. / (np.sqrt((D * u) ** 2 + ep)), 0, n, n)
                    # Linearized diffusion matrix, also approximation of Hessian.
                    L = DT * Q * D
                elif diffkernel == 'sq':
                    L = DT * D
                else:
                    raise ValueError('Invalid diffkernel value')

                # Gradient of functional.
                g = AT(A(u)) - ATd
                g = g + alph * L * u
                # Build preconditioner.
                if precondflag:
                    c = np.cumsum(range(n, 0, -1))
                    B = alph * L + sparse.spdiags(c[::-1], 0, n, n)
                    # droptol = 1.0e-2
                    R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
                    P = np.dot(R.transpose(), R)
                else:
                    P = None

                # Prepare to solve linear equation.

                def linop(v):
                    return (alph * L * v + AT(A(v)))

                linop = splin.LinearOperator((n, n), linop)

                s, info_i = sparse.linalg.cg(
                    linop, -g, x0=None, tol=cgtol, maxiter=cgmaxit, callback=None,
                    M=P, atol='legacy')
                if diagflag:
                    self.log_iteration(ii, s[0], u, g)
                    if (info_i > 0):
                        logging.warning(
                            "WARNING - convergence to tolerance not achieved!")
                    elif (info_i < 0):
                        logging.warning("WARNING - illegal input or breakdown")

                # Update current solution
                u = u + s
                smoothed_signal = np.cumsum(u) * dx

                offset = np.mean(data) - np.mean(smoothed_signal)
                smoothed_signal = smoothed_signal + offset
                # Display plot
                if plotflag:
                    plt.plot(u / dx)
                    plt.show()

            u = u / dx

        return smoothed_signal, u

    def _obj_function(self, alpha, signal):
        """Internal method. Objective function for wavelet scales optimization
        It requires to know the analytical expression of the Equations. Only available for synth. data

        Args:
            scale (float): optimization variable
        Returns:
            float: Least squares error.
        """
        print(signal.shape)
        x_dot_wt = np.zeros_like(self.x_train_noise)
        dt = self.time_array[1] - self.time_array[0]
        # Compute the continuous wavelet transform with the given scales
        y, dYdX = self.TVRegDiff(signal, 50, alpha, .01, dx=dt, ep=1e-6, plotflag=False, precondflag=True, diffkernel=('sq'))

        # Calculate the difference between the transformed signal and the reference signal
        difference = self.x_train_noise - y

        return np.sum(difference ** 2)

    def optimize_scale(self):

        initial_value = 1
        optimization = minimize(self._obj_function, initial_value, args= self.x_train_noise[:, 0],
                                method = 'Nelder-Mead')

        return optimization.x


