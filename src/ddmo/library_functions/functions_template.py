# At the top level of your module
def identity(x):
    return x

def poly_interaction(x, y, opt_vars):
    return x ** opt_vars[0] * y ** opt_vars[1]

def sine(x):
    return np.sin(x)

# Refactor get_base_functions
def get_base_functions():
    return [identity, square, sine]
