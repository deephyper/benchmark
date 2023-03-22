import os

import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob

nb_dim = os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5)
domain = (-32.768, 32.768)
problem = HpProblem()
for i in range(nb_dim):
    problem.add_hyperparameter(domain, f"x{i}")


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
