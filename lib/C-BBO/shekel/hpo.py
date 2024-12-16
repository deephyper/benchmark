import os
import time

import numpy as np
from deephyper.evaluator import RunningJob, profile
from deephyper.hpo import HpProblem

DEEPHYPER_BENCHMARK_NDIMS = 4
domain = (0.0, 10.0)
problem = HpProblem()
for i in range(DEEPHYPER_BENCHMARK_NDIMS):
    problem.add_hyperparameter(domain, f"x{i}")


def shekel(x):
    m = 10
    beta = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]).T
    C = np.array(
        [
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        ]
    )
    return -sum(
        [
            1 / (np.sum((x - C[:, i]) ** 2) + beta[i])
            for i in range(m)
        ]
    )


@profile
def run(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -shekel(x)


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    config = {f"x{i}": 4.0 for i in range(DEEPHYPER_BENCHMARK_NDIMS)}
    print(f"{config=}")
    result = run(RunningJob(parameters=config))
    print(f"{result=}")
