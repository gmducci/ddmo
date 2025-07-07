# ddmo


`ddmo` is a data-driven model optimization package tailored for identifying mathematical models of ordinary differential equations from data.

<!-- To do -->
# To do
## Refactoring process to-do list
- [ ] Remove `ddmo/ddmo` folder. The source code stays in `src`
- [ ] `src/ddmo` remove all the main files. The main files stay in `./examples` folder.


## High priority
- [ ] Set from the main file the objective functions and its number. The user should never tweak the `src/ddmo/optimization` folder to adjust these things.
- [ ] Uniform the library between ddmo and pysindy. Absolute priority :sparkles:
- [x] Implement the fit intercept in the main branch.
- [ ] Remove the optimal design folder & related files :fire:.

## Low priority

- [ ] Add Documentation
- [ ] Add additional templates w/ Examples
- [ ] Add LICENSE

See also the [open issues](https://thgitlab.rz-berlin.mpg.de/ducci/ddmo/-/issues) for a full list of proposed features (and known issues).


# DDMO

**Data-Driven Model Optimization (DDMO)** is a Python library designed for the discovery of dynamical systems from noisy experimental data using sparse regression techniques and multi-objective optimization. DDMO provides a modular framework where different aspects (such as data differentiation, custom ODE equations, user-defined function libraries, and optimization routines) are clearly separated, making it easy to extend and customize.

## Features

- **Differentiation Module:**  
  Provides methods for smoothing, noise addition, and derivative estimation through Kalman smoothing, wavelet-based methods, and Total Variation Regularization.

- **Equations Module:**  
  Contains pre-defined ODE models (e.g. Bimolecular reactions, polynomial ODEs) for testing and validation of the optimization process.

- **Library Functions Module:**  
  Implements a framework to create custom libraries of candidate functions (e.g., polynomial functions with user-defined exponents), which can be easily extended by the user.

- **Optimizer Module:**  
  Uses multi-objective optimization (based on pymoo) along with sparse regression (via scikit-learn) to identify the underlying differential equations from data.

- **Modular and Extensible:**  
  The package is designed to be user-friendly and simple to extend; adding new functions or models is straightforward.

## Installation

<!-- Requirements:
- Python 3.7+
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PySINDy](https://github.com/dynamicslab/pysindy)
- [pymoo](https://pymoo.org/)
- [pynumdiff](https://github.com/sclever/pynumdiff)
- [statsmodels](https://www.statsmodels.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/) -->

Clone the repository and install the package using pip:

```bash
git clone https://github.com/FHI-theory/ddmo.git
cd ddmo
pip install .
```

## Running examples

Just run the following
`python3 examples/example_generator_standalone.py`


<!-- CONTRIBUTING -->
## Contributing

Contributions to the project are **greatly appreciated**.

Should you have features in mind, please fork the repo and create a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourFeature`)
3. Commit your Changes (`git commit -m 'Add some feature'`)
4. Push to the Branch (`git push origin feature/YourFeature`)
5. Open a Pull Request
