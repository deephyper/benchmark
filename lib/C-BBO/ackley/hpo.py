import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob


DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))
DEEPHYPER_BENCHMARK_OFFSET = float(os.environ.get("DEEPHYPER_BENCHMARK_OFFSET", 4.0))
DEEPHYPER_BENCHMARK_NDIMS_SLACK = int(
    os.environ.get("DEEPHYPER_BENCHMARK_NDIMS_SLACK", 0)
)

# The original range is simetric (-32.768, 32.768) but we make it less simetric to avoid
# Grid sampling or QMC sampling to directly hit the optimum...
domain = (-32.768 - DEEPHYPER_BENCHMARK_OFFSET, 32.768 - DEEPHYPER_BENCHMARK_OFFSET)
problem = HpProblem()
for i in range(DEEPHYPER_BENCHMARK_NDIMS - DEEPHYPER_BENCHMARK_NDIMS_SLACK):
    problem.add_hyperparameter(domain, f"x{i}")

# Add slack/dummy dimensions (useful to test predicors which are sensitive to unimportant features)
for i in range(
    DEEPHYPER_BENCHMARK_NDIMS - DEEPHYPER_BENCHMARK_NDIMS_SLACK,
    DEEPHYPER_BENCHMARK_NDIMS,
):
    problem.add_hyperparameter(domain, f"z{i}")


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    y = term1 + term2 + a + np.exp(1)
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

    return -ackley(x)


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
