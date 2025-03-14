"""Module defining the problem and run-function of the benchmark."""

import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob
from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


def michal(x, m=10):
    ix2 = np.arange(1, len(x) + 1) * x**2
    y = -np.sum(np.sin(x) * np.power(np.sin(ix2 / np.pi), 2 * m))
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

    return -michal(x)


class MichalHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Michal problem."""

    def __init__(
        self,
        p_num,
    ):
        assert p_num in [2, 5, 10]
        self.p_num = p_num
        if self.p_num == 2:
            self.x_max = np.array([2.20, 1.57])
            self.y_max = 1.8013
        elif self.p_num == 5:
            self.y_max = 4.687658
        elif self.p_num == 10:
            self.y_max = 9.66015

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


class MichalHPOBenchmark(HPOBenchmark):
    def refresh_settings(self):
        self.DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))
        assert self.DEEPHYPER_BENCHMARK_NDIMS in [2, 5, 10]

    @property
    def problem(self):
        domain = (0.0, np.pi)
        problem = HpProblem()
        for i in range(self.DEEPHYPER_BENCHMARK_NDIMS):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):
        return run_function

    @property
    def scorer(self):
        return MichalHPOScorer(self.DEEPHYPER_BENCHMARK_NDIMS)


benchmark = MichalHPOBenchmark()
