import time
import scipy.stats
import numpy as np


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)


def run(config, sleep_duration=0, sleep_duration_noise=0):
    if sleep_duration > 0:
        t_sleep = max(
            0, scipy.stats.norm.rvs(sleep_duration, sleep_duration_noise, size=1)[0]
        )
        time.sleep(t_sleep)
    x = np.array([config[k] for k in config if "x" in k])
    return -ackley(x)
