"""Module defining the problem and run-function of the benchmark."""

import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob
from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


def cossin(x):
    x = np.sum(x)
    y = np.cos(5 * x / 10) + 2 * np.sin(x / 10) + x / 100
    return y


@profile
def run_function(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20, y_noise=0.0) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    y = cossin(x)

    if y_noise > 0.0:
        rng = np.random.RandomState(int(job.id.split(".")[-1]))
        eps = rng.normal(loc=0.0, scale=y_noise)
        y += eps

    return y


class CosSinHPOScorer(HPOScorer):
    """A class defining performance evaluators for the CosSin problem."""

    def __init__(
        self,
        p_num,
        p_num_slack,
    ):
        self.p_num = p_num
        self.x_min = np.full((self.p_num,), 18.6738)
        self.x_min[self.p_num - p_num_slack :] = np.nan
        self.y_min = 3.1

    def simple_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the regret of a list of given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        return self.y_min - y

    def cumul_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the cumulative regret of an array of ordered given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of cumulative regret values.
        """
        return np.cumsum(self.simple_regret(y))


class CosSinHPOBenchmark(HPOBenchmark):
    def refresh_settings(self):
        self.DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 1))
        self.DEEPHYPER_BENCHMARK_OFFSET = float(os.environ.get("DEEPHYPER_BENCHMARK_OFFSET", 0.0))
        self.DEEPHYPER_BENCHMARK_NDIMS_SLACK = int(
            os.environ.get("DEEPHYPER_BENCHMARK_NDIMS_SLACK", 0)
        )

    @property
    def problem(self):
        domain = (
            -50.0 - self.DEEPHYPER_BENCHMARK_OFFSET,
            50.0 - self.DEEPHYPER_BENCHMARK_OFFSET,
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
        return CosSinHPOScorer(self.DEEPHYPER_BENCHMARK_NDIMS, self.DEEPHYPER_BENCHMARK_NDIMS_SLACK)


benchmark = CosSinHPOBenchmark()
