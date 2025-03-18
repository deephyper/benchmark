"""here."""

import numpy as np
import time

from deephyper.evaluator import profile
from deephyper.evaluator import RunningJob
from deephyper.hpo import HpProblem


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    y = term1 + term2 + a + np.exp(1)
    return y


@profile
def run_function(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -ackley(x)


class AckleyBenchmark:
    def __init__(self, ndims=5, offset=4.0, slack=0) -> None:
        self.ndims = ndims
        self.offset = offset
        self.slack = slack

    @property
    def problem(self):
        # The original range is simetric (-32.768, 32.768) but we make it less simetric to avoid
        # Grid sampling or QMC sampling to directly hit the optimum...
        domain = (
            -32.768 - self.offset,
            32.768 - self.offset,
        )
        problem = HpProblem()
        for i in range(self.ndims - self.slack):
            problem.add_hyperparameter(domain, f"x{i}")

        # Add slack/dummy dimensions (useful to test predicors which are sensitive
        # to unimportant features)
        for i in range(
            self.ndims - self.slack,
            self.ndims,
        ):
            problem.add_hyperparameter(domain, f"z{i}")
        return problem

    @property
    def run_function(self):
        return run_function
