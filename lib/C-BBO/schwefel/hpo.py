import os
import time

import numpy as np
from deephyper.evaluator import RunningJob, profile
from deephyper.hpo import HpProblem

DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))
domain = (-500.0, 500.0)
problem = HpProblem()
for i in range(DEEPHYPER_BENCHMARK_NDIMS):
    problem.add_hyperparameter(domain, f"x{i}")


def schwefel(x):  # schw.m
    n = len(x)
    return 418.9829 * n - sum(x * np.sin(np.sqrt(np.abs(x))))


@profile
def run(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -schwefel(x)


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
