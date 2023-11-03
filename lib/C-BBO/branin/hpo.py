import os
import time

import numpy as np

from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob


problem = HpProblem()
problem.add_hyperparameter((-5.0, 10.0), f"x0")
problem.add_hyperparameter((0.0, 15.0), f"x1")


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
def run(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -branin(x)


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    default_config = {"x0": -np.pi, "x1": 12.275}  # sol
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
