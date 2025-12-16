class Brussellator:
    def __init__(self, a=10, b=8 / 3, alpha=2, beta=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta

    def dynamics(self, x_0, t):
        x, y = x_0
        dxdt = 1 - (self.b + 1) * x + self.a * ((x**self.alpha) * (y**self.beta))
        dydt = self.b * x - self.a * ((x**self.alpha) * (y**self.beta))
        return [dxdt, dydt]
