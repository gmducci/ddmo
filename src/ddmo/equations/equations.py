import numpy as np


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
       # dx_dt = 1/(1+z**7) - \
       #         self.k2 * x
        dy_dt = self.k3 * x - self.k4 * y
        dz_dt = self.k5 * y - self.k6 * z

        vel_array = np.array([dx_dt, dy_dt, dz_dt], float)
        return vel_array

class Lorenz:
    def __init__(self, sigma=10., r=28, b=8/3):

        self.sigma = sigma
        self.r = r
        self.b = b
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
        dxdt = self.sigma*(y-x)
        dydt = x*(self.r-z)-y
        dzdt = x*y - self.b*z
        vel_array = np.array([dxdt, dydt, dzdt], float)
        return vel_array

class RedoxModel:
    def __init__(self, a=1000, b=2, c=10000, d=0.2, e=0.1, q=0.1, p=1):

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.q = q
        self.p = p
        self.dimension = 4
    def dynamics(self, x0, t):

        """ODE system
        This function will be passed to the numerical integrator

        Inputs:
            x0: initial values
            t: time

        Outputs:
            x_dot: velocity vector
        """
        D1, D2, R, A = x0
        dD1_dt = self.p - self.a*A*D1 - self.d*D1
        dD2_dt = self.d*D1 - self.e*D2
        dR_dt = self.e*D2 - self.q*R
        dA_dt = self.b*(1-A)*R - self.a*A*D1

        vel_array = np.array([dD1_dt, dD2_dt, dR_dt, dA_dt], float)
        return vel_array

class Selkov:

    def __init__(self, a = 1.1, b=1, gamma = 2.1):

        self.a = a
        self.b = b
        self.gamma = gamma
        self.dimension = 2
    def dynamics(self, x0, t):

        """ODE system
        This function will be passed to the numerical integrator

        Inputs:
            x0: initial values
            t: time

        Outputs:
            x_dot: velocity vector
        """
        x, y = x0
        dx_dt = 1 - x * (y ** self.gamma)
        dy_dt = self.a*x*y**(self.gamma) - self.a*y

        vel_array = np.array([dx_dt, dy_dt], float)
        return vel_array



class DelOlmoOscillator:

    def __init__(self,

                 # default, AND funnel
                 b0=0.25, t1=1.0, t2=0.5, q1=.25,
                 dx=0.5, dy=0.2, dz1=0.05, dz2=0.5,
                 dby=0.0, dbz1=0.0, dbz2=0.,
                 V=1.0, K=1, h=8, K2=1, h2=8,
                 Kx=1.0, Ky=.25, Kz1=2.50, Kz2=0.25):
        self.b0 = b0  # translation rate
        self.dx = dx  # degradation x
        self.dy = dy  # degradation y
        self.dz1 = dz1  # degradation dz1
        self.dz2 = dz2  # degradation dz1
        self.t1 = t1  # nuc import rate
        self.t2 = t2  # nuc import rate
        self.q1 = q1  # maturation of CRY in nuc
        self.V = V
        self.K = K
        self.K2 = K2
        self.h = h
        self.h2 = h2
        self.Kx = Kx
        self.Ky = Ky
        self.Kz1 = Kz1
        self.Kz2 = Kz2
        self.dby = dby  # basal degradation
        self.dbz1 = dbz1
        self.dbz2 = dbz2
        self.dimension = 4

    def dynamics(self, Y0, t):
        """
        """
        f = np.zeros_like(Y0)
        x = Y0[0::4]
        y = Y0[1::4]
        z1 = Y0[2::4]
        z2 = Y0[3::4]

        fx = f[0::4]
        fy = f[1::4]
        fz1 = f[2::4]
        fz2 = f[3::4]

        # odes
        # with AND funnel
        fx[:] = (self.V) / (self.K ** self.h + z1 ** self.h) * \
                (1 / (self.K2 ** self.h2 + z2 ** self.h2)) - \
                self.dx * x / (self.Kx + x)
        fy[:] = self.b0 * x - (self.t1 + self.t2 + \
                               self.dy / (self.Ky + y + z2 + z1) + self.dby) * y
        fz1[:] = self.t1 * y - (self.q1 + \
                                self.dz1 / (self.Kz1 + z1 + z2 + y) + \
                                self.dbz1) * z1
        fz2[:] = self.q1 * z1 + self.t2 * y - (self.dz2 / (self.Kz2 + z1 + y + z2) + \
                                               self.dbz2) * z2

        return f
