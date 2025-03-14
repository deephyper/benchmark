"""Module defining the problem and run-function of the benchmark."""

import os
import time

import numpy as np
from deephyper.evaluator import RunningJob, profile
from deephyper.hpo import HpProblem
from scipy.optimize import rosen

from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


@profile
def run_function(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -rosen(x)


class RosenHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Rosen problem."""

    def __init__(self, p_num):
        self.p_num = p_num
        self.x_max = np.ones(self.p_num)
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


class RosenHPOBenchmark(HPOBenchmark):
    def refresh_settings(self):
        self.DEEPHYPER_BENCHMARK_NDIMS = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))

    @property
    def problem(self):
        domain = (-5.0, 10.0)
        problem = HpProblem()
        for i in range(self.DEEPHYPER_BENCHMARK_NDIMS):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):
        return run_function

    @property
    def scorer(self):
        return RosenHPOScorer(self.DEEPHYPER_BENCHMARK_NDIMS)


benchmark = RosenHPOBenchmark()
