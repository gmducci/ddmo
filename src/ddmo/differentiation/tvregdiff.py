"""
Python function to estimate derivatives from noisy data based on
Rick Chartrand's Total Variation Regularized Numerical
Differentiation (TVDiff) algorithm.

Example:
>>> u = TVRegDiff(data, iter, alph, u0, scale, ep, dx,
...               plotflag, diagflag)

Rick Chartrand (rickc@lanl.gov), Apr. 10, 2011
Please cite Rick Chartrand, "Numerical differentiation of noisy,
nonsmooth data," ISRN Applied Mathematics, Vol. 2011, Article ID 164564,
2011.

Copyright notice:
Copyright 2010. Los Alamos National Security, LLC. This material
was produced under U.S. Government contract DE-AC52-06NA25396 for
Los Alamos National Laboratory, which is operated by Los Alamos
National Security, LLC, for the U.S. Department of Energy. The
Government is granted for, itself and others acting on its
behalf, a paid-up, nonexclusive, irrevocable worldwide license in
this material to reproduce, prepare derivative works, and perform
publicly and display publicly. Beginning five (5) years after
(March 31, 2011) permission to assert copyright was obtained,
subject to additional five-year worldwide renewals, the
Government is granted for itself and others acting on its behalf
a paid-up, nonexclusive, irrevocable worldwide license in this
material to reproduce, prepare derivative works, distribute
copies to the public, perform publicly and display publicly, and
to permit others to do so. NEITHER THE UNITED STATES NOR THE
UNITED STATES DEPARTMENT OF ENERGY, NOR LOS ALAMOS NATIONAL
SECURITY, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR
REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
RIGHTS.

BSD License notice:
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

     Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     disclaimer.
     Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials
     provided with the distribution.
     Neither the name of Los Alamos National Security nor the names of its
     contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

#########################################################
#                                                       #
# Python translation by Simone Sturniolo                #
# Rutherford Appleton Laboratory, STFC, UK (2014)       #
# simonesturniolo@gmail.com                             #
#                                                       #
#########################################################
"""


import logging

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splin

_has_matplotlib = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    _has_matplotlib = False
    logging.warning("Matplotlib is not installed - plotting "
                    "functionality disabled")


def log_iteration(ii, s0, u, g):
    relative_change = np.linalg.norm(s0) / np.linalg.norm(u)
    g_norm = np.linalg.norm(g)
    logging.info(f'iteration {ii:4d}: relative change = {relative_change:.3e}, '
                 f'gradient norm = {g_norm:.3e}\n')


def TVRegDiff(data, itern, alph, u0=None, scale='small', ep=1e-6, dx=None,
              plotflag=_has_matplotlib, diagflag=True, precondflag=True,
              diffkernel='abs', cgtol=1e-4, cgmaxit=100):
    """
    Estimate derivatives from noisy data based using the Total
    Variation Regularized Numerical Differentiation (TVDiff)
    algorithm.

    Parameters
    ----------
    data : ndarray
        One-dimensional array containing series data to be
        differentiated.
    itern : int
        Number of iterations to run the main loop.  A stopping
        condition based on the norm of the gradient vector g
        below would be an easy modification.  No default value.
    alph : float
        Regularization parameter.  This is the main parameter
        to fiddle with.  Start by varying by orders of
        magnitude until reasonable results are obtained.  A
        value to the nearest power of 10 is usally adequate.
        No default value.  Higher values increase
        regularization strenght and improve conditioning.
    u0 : ndarray, optional
        Initialization of the iteration.  Default value is the
        naive derivative (without scaling), of appropriate
        length (this being different for the two methods).
        Although the solution is theoretically independent of
        the initialization, a poor choice can exacerbate
        conditioning issues when the linear system is solved.
    scale : {large' or 'small' (case insensitive)}, str, optional
        Default is 'small'.  'small' has somewhat better boundary
        behavior, but becomes unwieldly for data larger than
        1000 entries or so.  'large' has simpler numerics but
        is more efficient for large-scale problems.  'large' is
        more readily modified for higher-order derivatives,
        since the implicit differentiation matrix is square.
    ep : float, optional
        Parameter for avoiding division by zero.  Default value
        is 1e-6.  Results should not be very sensitive to the
        value.  Larger values improve conditioning and
        therefore speed, while smaller values give more
        accurate results with sharper jumps.
    dx : float, optional
        Grid spacing, used in the definition of the derivative
        operators.  Default is the reciprocal of the data size.
    plotflag : bool, optional
        Flag whether to display plot at each iteration.
        Default is True.  Useful, but adds significant
        running time.
    diagflag : bool, optional
        Flag whether to display diagnostics at each
        iteration.  Default is True.  Useful for diagnosing
        preconditioning problems.  When tolerance is not met,
        an early iterate being best is more worrying than a
        large relative residual.
    precondflag: bool, optional
        Flag whether to use a preconditioner for conjugate gradient solution.
        Default is True. While in principle it should speed things up,
        sometimes the preconditioner can cause convergence problems instead,
        and should be turned off. Note that this mostly makes sense for 'small'
        scale problems; for 'large' ones, the improved preconditioner is one
        of the main features of the algorithms and turning it off defeats the
        point.
    diffkernel: str, optional
        Kernel to use in the integral to smooth the derivative. By default it's
        the absolute value, |u'| (value: "abs"). However, it can be changed to
        being the square, (u')^2 (value: "sq"). The latter produces smoother
        derivatives, whereas the absolute values tends to make them more blocky.
        Default is abs.
    cgtol: float, optional
        Tolerance to use in conjugate gradient optimisation. Default is 1e-4.
    cgmaxit: int, optional
        Maximum number of iterations to use in conjugate gradient optimisation.
        Default is 100


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

    # Default checking. (u0 is done separately within each method.)
    if dx is None:
        dx = 1.0 / n

    # Different methods for small- and large-scale problems.
    if (scale.lower() == 'small'):

        # Differentiation operator
        d0 = -np.ones(n)/dx
        du = np.ones(n-1)/dx
        dl = np.zeros(n-1)
        dl[-1] = d0[-1]
        d0[-1] *= -1

        D = sparse.diags([dl, d0, du], [-1, 0, 1])
        DT = D.transpose()

        # Antidifferentiation and its adjoint
        def A(x): return (np.cumsum(x) - 0.5 * (x + x[0])) * dx

        def AT(x): return np.concatenate([[sum(x[1:])/2.0],
                                          (sum(x)-np.cumsum(x)+0.5*x)[1:]])*dx

        # Default initialization is naive derivative

        if u0 is None:
            u0 = D*data

        u = u0.copy()
        # Since Au( 0 ) = 0, we need to adjust.
        ofst = data[0]
        # Precompute.
        ATb = AT(ofst - data)        # input: size n

        smoothed_signal = data.copy()  # Initialize smoothed signal

        # Main loop.
        for ii in range(1, itern+1):
            if diffkernel == 'abs':
                # Diagonal matrix of weights, for linearizing E-L equation.
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
                # Linearized diffusion matrix, also approximation of Hessian.
                L = dx * DT * Q * D
            elif diffkernel == 'sq':
                L = dx * DT * D
            else:
                raise ValueError('Invalid diffkernel value')

            # Gradient of functional.
            g = AT(A(u)) + ATb + alph * L * u

            # Prepare to solve linear equation.
            if precondflag:
                # Simple preconditioner.
                P = alph * sparse.spdiags(L.diagonal() + 1, 0, n, n)
            else:
                P = None

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(
                linop, g, x0=None, rtol=cgtol, atol=0, maxiter=cgmaxit,
                callback=None, M=P)

            if diagflag:
                log_iteration(ii, s[0], u, g)
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
        def A(v): return np.cumsum(v)

        def AT(w): return (sum(w) * np.ones(len(w)) -
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
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
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

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(
                linop, -g, x0=None, tol=cgtol, maxiter=cgmaxit, callback=None,
                M=P, atol='legacy')
            if diagflag:
                log_iteration(ii, s[0], u, g)
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



if __name__ == "__main__":

    # Command line operation
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import odeint

    matplotlib.use('Qt5Agg')
    import pylops


    # Goodwin class
    class GoodwinModel:
        def __init__(self, k1=1, k2=0.2, k3=1, k4=0.15, k5=1, k6=0.1,
                     K1=1, K2=1, K4=1, K6=1, n=9.5, perturbation=False):
            self.k1 = k1
            self.k2 = k2
            self.k3 = k3
            self.k4 = k4
            self.k5 = k5
            self.k6 = k6
            self.K1 = K1
            self.K2 = K2
            self.K4 = K4
            self.K6 = K6
            self.n = n
            self.perturbation = perturbation
            self.dimension = 3

        def dynamics(self, x0, t):
            """ODE system
            This function will be passed to the numerical integrator

            Inputs:
                x0: initial values
                t: time

            Outputs:
                x_dot: velocity vector
            """

            x, y, z = x0

            dx_dt = self.k1 * (self.K1 ** self.n / (self.K1 ** self.n + z ** self.n)) - \
                    self.k2 * x

            dy_dt = self.k3 * x - self.k4 * y

            dz_dt = self.k5 * y - self.k6 * z

            vel_array = np.array([dx_dt, dy_dt, dz_dt], float)
            return vel_array


    # Routine to add noise on the clean data
    def add_noise(x_train, time_array, noise_level):
        x_train_noise = np.zeros_like(x_train)

        for i in range(len(x_train[0, :])):
            x_train_noise[:, i] = x_train[:, i] + \
                                  np.random.normal(0, x_train[:, i].std(), time_array.size) * noise_level

        return x_train_noise


    # Initialization of the Goodwin object
    GoodwinObj = GoodwinModel(n=9.5)
    goodwin_eqs = GoodwinObj.dynamics
    time_steps = 50000
    time = np.linspace(0, 500, time_steps)
    x_0 = [0, 1, 1]
    x_1 = [.1, .5, .1]

    # numerical integration of the Goodwin model
    solution = odeint(goodwin_eqs, x_0, time)
    solution_1 = odeint(goodwin_eqs, x_1, time)
    # Adding gaussian noise to the solution
    noise_level = 0.5
    x_noise = add_noise(solution, time, noise_level)
    x_noise_1 = add_noise(solution_1, time, noise_level)
    # calculating the ground truth derivatives
    true_derivatives = np.zeros_like(x_noise)
    td_1 = np.zeros_like(x_noise)
    td_noiseless = np.zeros_like(x_noise)

    for i in range(len(time)):
        td_noiseless[i, :] = goodwin_eqs(solution_1[i, :], time[i])
        true_derivatives[i, :] = goodwin_eqs(x_noise[i, :], time[i])
        td_1[i, :] = goodwin_eqs(x_noise_1[i, :], time[i])



    dX = time[1] - time[0]
    Sop = pylops.Smoothing1D(nsmooth=1, dims=(len(time)), dtype="float32")

    y0 = Sop * x_noise[:, 0]
    y1 = Sop * x_noise[:, 1]
    y2 = Sop * x_noise[:, 2]
    x_hat = np.array([y0, y1, y2]).T

    y, dYdX = TVRegDiff(x_noise_1[:, 2], 100, 10, dx=dX, ep=1e-6,
                     scale=('small'),
                     plotflag=False, precondflag=True,
                     diffkernel=('sq'))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    off = np.mean(x_noise_1[:, 2]) - np.mean(y)
    # First panel
    axs[0].plot(time, x_noise_1[:, 2], label='Noisy Data')
    axs[0].plot(time, solution_1[:, 2], label='GT', lw = 2, color = 'red')
    axs[0].plot(time, y+off, label='Smoothed Signal')
    axs[0].set_title('Noisy Data and Smoothed Signal')
    axs[0].legend()

    # Second panel
    # axs[1].plot(time, td_1[:, 0], label='Derivative of Noisy Data')
    axs[1].plot(time, td_noiseless[:, 2], lw=2, color='red', label='Derivative of Smoothed Signal')
    axs[1].plot(time, dYdX, label='dYdX')
    axs[1].set_title('Derivatives')
    axs[1].legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
