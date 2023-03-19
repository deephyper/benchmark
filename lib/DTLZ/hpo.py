import os

import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob
from dtlz import dtlz2 as DTLZ

# Set problem dims (or read from ENV)
nb_dim = os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5)
nb_obj = os.environ.get("DEEPHYPER_BENCHMARK_NOBJS", 2)
domain = (0., 1.)
soln_offset = 0.6

# Set up problem
problem = HpProblem()
dtlz_obj = DTLZ(nb_dim, nb_obj, offset=soln_offset)
for i in range(nb_dim):
    problem.add_hyperparameter(domain, f"x{i}")


@profile
def run(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:

    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    ff = [fi for fi in dtlz_obj(x)]

    return ff


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
