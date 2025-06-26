import numpy as np

class OneDProblem:
    def __init__(self, a=0.1, b=0.5, c=0.5):
        self.a = a
        self.b = b
        self.c = c

    def dynamics(self, x_0, t):
        dxdt = self.a + self.b * x_0 + self.c * x_0**2
        return dxdt



class Bimolecular:
    def __init__(self, k=0.1, b0=1):
        self.k = k
        self.b0 = b0

    def dynamics(self, x_0, t):
        dxdt = self.b0 * self.k * x_0 - self.k * x_0**2
        return dxdt

    def dynamics_2(self, x_0, t):

        x, y = x_0
        dxdt = -self.k * x * y
        dydt = -self.k * x * y

        return np.asarray([dxdt, dydt])
