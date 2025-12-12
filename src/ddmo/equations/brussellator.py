class Brussellator:
    def __init__(self, a=10, b=8 / 3, c=28):
        self.a = a
        self.b = b
        self.c = c

    def dynamics(self, x_0, t):
        x, y = x_0
        dxdt = 1 - (self.b + 1) * x + self.a * ((x**2) * y)
        dydt = self.b * x - self.a * ((x**2) * y)
        return [dxdt, dydt]
