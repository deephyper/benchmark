import numpy as np
from deephyper.evaluator import profile


def branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    x1 = x[0]
    x2 = x[1]
    term1 = a * np.square(x2 - b*np.square(x1) + c*x1 - r)
    term2 = s*(1-t)*np.cos(x1)
    y = term1 + term2 + s + 5*x1
    return y

@profile
def run(config):
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -branin(x)