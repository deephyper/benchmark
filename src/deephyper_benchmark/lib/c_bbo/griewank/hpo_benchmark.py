"""Module defining the problem and run-function of the benchmark."""

import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob
from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


def griewank(x, fr=4000):
    n = len(x)
    j = np.arange(1.0, n + 1)
    s = np.sum(x**2)
    p = np.prod(np.cos(x / np.sqrt(j)))
    return s / fr - p + 1


@profile
def run_function(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -griewank(x)


class GriewankHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Griewank problem."""

    def __init__(
        self,
        p_num,
        p_num_slack,
        offset=0,
    ):
        self.p_num = p_num
        self.x_max = np.full(self.p_num, fill_value=-offset)
        self.x_max[p_num - p_num_slack :] = np.nan
        self.y_max = 0.0

    def simple_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the regret of a list of given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        return self.y_max - y

    def cumul_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the cumulative regret of an array of ordered given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of cumulative regret values.
        """
        return np.cumsum(self.simple_regret(y))


class GriewankHPOBenchmark(HPOBenchmark):
    def refresh_settings(self):
        self.DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))
        self.DEEPHYPER_BENCHMARK_OFFSET = float(os.environ.get("DEEPHYPER_BENCHMARK_OFFSET", 4.0))
        self.DEEPHYPER_BENCHMARK_NDIMS_SLACK = int(
            os.environ.get("DEEPHYPER_BENCHMARK_NDIMS_SLACK", 0)
        )

    @property
    def problem(self):
        domain = (
            -600.0 - self.DEEPHYPER_BENCHMARK_OFFSET,
            600.0 - self.DEEPHYPER_BENCHMARK_OFFSET,
        )
        problem = HpProblem()
        for i in range(self.DEEPHYPER_BENCHMARK_NDIMS - self.DEEPHYPER_BENCHMARK_NDIMS_SLACK):
            problem.add_hyperparameter(domain, f"x{i}")

        # Add slack/dummy dimensions (useful to test predicors which are sensitive
        # to unimportant features)
        for i in range(
            self.DEEPHYPER_BENCHMARK_NDIMS - self.DEEPHYPER_BENCHMARK_NDIMS_SLACK,
            self.DEEPHYPER_BENCHMARK_NDIMS,
        ):
            problem.add_hyperparameter(domain, f"z{i}")
        return problem

    @property
    def run_function(self):
        return run_function

    @property
    def scorer(self):
        return GriewankHPOScorer(
            self.DEEPHYPER_BENCHMARK_NDIMS,
            self.DEEPHYPER_BENCHMARK_NDIMS_SLACK,
            self.DEEPHYPER_BENCHMARK_OFFSET,
        )


benchmark = GriewankHPOBenchmark()
