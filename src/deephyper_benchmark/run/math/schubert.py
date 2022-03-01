import numpy as np
from deephyper.evaluator import profile


def schubert(x):
    x1 = x[0]
    x2 = x[1]
    ii = np.arange(1, 6)
    sum1 = np.sum(np.multiply(ii, np.cos((ii+1)*x1+ii)))
    sum2 = np.sum(np.multiply(ii, np.cos((ii+1)*x2+ii)))
    y = sum1 * sum2
    return y

@profile
def run(config):
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -schubert(x)