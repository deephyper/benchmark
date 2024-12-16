import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob


DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 2))
DEEPHYPER_BENCHMARK_OFFSET = float(os.environ.get("DEEPHYPER_BENCHMARK_OFFSET", 20.0))

# The original range is simetric (-100.0, 100.0) but we make it less simetric to avoid
# Grid sampling or QMC sampling to directly hit the optimum...
domain = (-100.0 - DEEPHYPER_BENCHMARK_OFFSET, 100.0 - DEEPHYPER_BENCHMARK_OFFSET)
problem = HpProblem()
for i in range(DEEPHYPER_BENCHMARK_NDIMS):
    problem.add_hyperparameter(domain, f"x{i}")


def easom(x):
    assert len(x) == 2
    y = (
        -np.cos(x[0])
        * np.cos(x[1])
        * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
    )
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

    return -easom(x)


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    default_config = {"x0": np.pi, "x1": np.pi}  # sol
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
