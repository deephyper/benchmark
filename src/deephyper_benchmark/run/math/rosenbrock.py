import numpy as np
from deephyper.evaluator import profile


def rosenbrock(x):
    xbar = 15*x - 5
    xibar = xbar[0:2]
    xnextbar = xbar[1:3]
    sum1 = np.sum(100*np.square(xnextbar-np.square(xibar)) + np.square(1 - xibar))
    y = (sum1 - 3.827*1e5) / (3.755*1e5)
    return y

@profile
def run(config):
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -rosenbrock(x)