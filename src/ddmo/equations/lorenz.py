
class Lorenz:
    def __init__(self, sigma=10, beta=8/3, rho=28):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

    def dynamics(self, x_0, t):
        x, y, z = x_0
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]
