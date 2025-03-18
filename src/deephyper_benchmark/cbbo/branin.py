"""here."""

import numpy as np
import time

from deephyper.evaluator import RunningJob
from deephyper.evaluator import profile
from deephyper.hpo import HpProblem


def branin(x):
    assert len(x) == 2
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
    return y


@profile
def run_function(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -branin(x)


class BraninBenchmark:
    @property
    def problem(self):
        problem = HpProblem()
        problem.add_hyperparameter((-5.0, 10.0), "x0")
        problem.add_hyperparameter((0.0, 15.0), "x1")
        return problem

    @property
    def run_function(self):
        return run_function
