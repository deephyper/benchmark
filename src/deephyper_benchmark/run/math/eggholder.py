import numpy as np
from deephyper.evaluator import profile


def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2+47) * np.sin(np.sqrt(np.abs(x2+x1/2+47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1-(x2+47))))
    y = term1 + term2
    return y

@profile
def run(config):
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -eggholder(x)