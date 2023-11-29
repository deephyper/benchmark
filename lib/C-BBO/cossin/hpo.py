import os
import time

import numpy as np

from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob


DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 1))
DEEPHYPER_BENCHMARK_NDIMS_SLACK = int(
    os.environ.get("DEEPHYPER_BENCHMARK_NDIMS_SLACK", 0)
)

# The original range is simetric (-32.768, 32.768) but we make it less simetric to avoid
# Grid sampling or QMC sampling to directly hit the optimum...
domain = (-50.0, 50.0)
problem = HpProblem()
for i in range(DEEPHYPER_BENCHMARK_NDIMS - DEEPHYPER_BENCHMARK_NDIMS_SLACK):
    problem.add_hyperparameter(domain, f"x{i}")

# Add slack/dummy dimensions (useful to test predicors which are sensitive to unimportant features)
for i in range(
    DEEPHYPER_BENCHMARK_NDIMS - DEEPHYPER_BENCHMARK_NDIMS_SLACK,
    DEEPHYPER_BENCHMARK_NDIMS,
):
    problem.add_hyperparameter(domain, f"z{i}")


def cossin(x):
    x = np.sum(x)
    y = np.cos(5 * x / 10) + 2 * np.sin(x / 10) + x / 100
    return y


@profile
def run(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20, noise=0.0) -> dict:
    job_id = int(job.id.split(".")[-1])
    config = job.parameters

    if sleep:
        rng = np.random.RandomState(job_id)
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    y = cossin(x)

    if noise > 0.0:
        rng = np.random.RandomState(job_id)
        eps = rng.normal(loc=0.0, scale=noise)
        y += eps

    return -y


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
