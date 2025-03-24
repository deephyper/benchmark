import os
import time

import numpy as np
from deephyper.evaluator import RunningJob, profile
from deephyper.hpo import HpProblem

from . import model as dtlz

# Read DTLZ problem name and acquire pointer
dtlz_prob = os.environ.get("DEEPHYPER_BENCHMARK_DTLZ_PROB", 2)
dtlz_prob_name = f"dtlz{dtlz_prob}"
dtlz_class_ptr = getattr(dtlz, dtlz_prob_name)

# Read problem dims and definition (or read from ENV)
nb_dim = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))
nb_obj = int(os.environ.get("DEEPHYPER_BENCHMARK_NOBJS", 2))
if dtlz_prob_name in ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5"]:
    soln_offset = float(os.environ.get("DEEPHYPER_BENCHMARK_DTLZ_OFFSET", 0.5))
elif dtlz_prob_name in ["dtlz6", "dtlz7"]:
    soln_offset = float(os.environ.get("DEEPHYPER_BENCHMARK_DTLZ_OFFSET", 0.0))
else:
    raise ValueError(f"Invalid problem {dtlz_prob_name}")
domain = (0.0, 1.0)

# Failures
DEEPHYPER_BENCHMARK_FAILURES = bool(int(os.environ.get("DEEPHYPER_BENCHMARK_FAILURES", 0)))

# Create problem
problem = HpProblem()
dtlz_obj = dtlz_class_ptr(nb_dim, nb_obj, offset=soln_offset)
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
    ff = [-fi for fi in dtlz_obj(x)]

    if DEEPHYPER_BENCHMARK_FAILURES:
        if any(xi < 0.25 for xi in x[nb_obj-1:]):
            ff = ["F" for _ in ff]

    return ff


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
