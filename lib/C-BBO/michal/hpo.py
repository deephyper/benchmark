import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob


DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 2))

domain = (0, np.pi)
problem = HpProblem()
for i in range(DEEPHYPER_BENCHMARK_NDIMS):
    problem.add_hyperparameter(domain, f"x{i}")


def michal(x, m=10):
    ix2 = np.arange(1, len(x) + 1) * x**2
    y = -np.sum(np.sin(x) * np.power(np.sin(ix2 / np.pi), 2 * m))
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

    return -michal(x)


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
