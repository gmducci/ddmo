import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

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
time_steps = 2000
time = np.linspace(0, 500, time_steps)
x_0 = [0, 1, 1]

# numerical integration of the Goodwin model
solution = odeint(goodwin_eqs, x_0, time)

# Adding gaussian noise to the solution
noise_level = 0.25
x_noise = add_noise(solution, time, noise_level)

# calculating the ground truth derivatives
true_derivatives = np.zeros_like(x_noise)
for i in range(len(time)):
    true_derivatives[i, :] = goodwin_eqs(x_noise[i, :], time[i])

# To save the files (uncomment below and adjust your path):
# np.save('x_noise.npy')
# np.save('true_derivatives.npy')


# Plotting routine, just to have an idea:

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot time vs x_noise
axs[0].plot(time, x_noise[:, 0], label='x')
axs[0].plot(time, x_noise[:, 1], label='y')
axs[0].plot(time, x_noise[:, 2], label='z')

axs[0].set_xlabel('Time')
axs[0].set_ylabel('x_noise')
axs[0].legend()
axs[0].set_title('Time vs x_noise')

# Plot time vs true derivatives
axs[1].plot(time, true_derivatives[:, 0], label=r'$\dot{x}$')
axs[1].plot(time, true_derivatives[:, 1], label=r'$\dot{y}$')
axs[1].plot(time, true_derivatives[:, 2], label=r'$\dot{z}$')

axs[1].set_xlabel('Time')
axs[1].set_ylabel('True Derivatives')
axs[1].legend()

# Add subtitles to each subplot
axs[0].set_title('Goodwin time series')
axs[1].set_title('Goodwin true derivatives')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show plot
plt.show()