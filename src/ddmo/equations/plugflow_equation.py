import numpy as np
class PlugFlow:
    def __init__(self, k0=0, k12=0.2, alpha = 2, beta = 3, v0 = .1):
        self.k0 = k0
        self.k12 = k12
        self.alpha = alpha
        self.beta = beta
        self.v0 = v0

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

        dx_dt = (-self.alpha * self.k12 * (x**self.alpha) * (y ** self.beta)) / self.v0
        dy_dt= -self.beta * self.k12 * (x**self.alpha) * (y ** self.beta) / self.v0
        dz_dt =  self.k12 * (x**self.alpha) * (y ** self.beta) / self.v0

        vel_array = np.array([dx_dt, dy_dt, dz_dt], float)
        return vel_array
