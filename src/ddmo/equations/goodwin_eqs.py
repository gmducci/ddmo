import numpy as np


class GoodwinModel:
    def __init__(
        self,
        k1=1,
        k2=0.2,
        k3=1,
        k4=0.15,
        k5=1,
        k6=0.1,
        K1=1,
        K2=1,
        K4=1,
        K6=1,
        n=9.5,
        perturbation=False,
    ):
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

        dx_dt = (
            self.k1 * (self.K1**self.n / (self.K1**self.n + z**self.n)) - self.k2 * x
        )

        dy_dt = self.k3 * x - self.k4 * y

        dz_dt = self.k5 * y - self.k6 * z

        vel_array = np.array([dx_dt, dy_dt, dz_dt], float)
        return vel_array
