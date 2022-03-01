import numpy as np
from deephyper.evaluator import profile


def bohachevsky(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    x1 = x[0]
    x2 = x[1]
    term1 = np.square(x1)
    term2 = 2*np.square(x2)
    term3 = -0.3 * np.cos(3*np.pi*x1) * np.cos(4*np.pi*x2)
    y = term1 + term2 + term3 + 0.3
    return y

@profile
def run(config):
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -bohachevsky(x)