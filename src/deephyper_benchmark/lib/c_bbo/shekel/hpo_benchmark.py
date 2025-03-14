"""Module defining the problem and run-function of the benchmark."""

import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob
from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


def shekel(x):
    m = 10
    beta = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]).T
    C = np.array(
        [
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        ]
    )
    return -sum([1 / (np.sum((x - C[:, i]) ** 2) + beta[i]) for i in range(m)])


@profile
def run_function(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -shekel(x)


class ShekelHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Shekel problem."""

    def __init__(self):
        self.p_num = 10
        self.x_max = np.full(self.p_num, fill_value=4.0)
        self.y_max = 10.5364

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


class ShekelHPOBenchmark(HPOBenchmark):
    @property
    def problem(self):
        domain = (0.0, 10.0)
        problem = HpProblem()
        for i in range(10):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):
        return run_function

    @property
    def scorer(self):
        return ShekelHPOScorer()


benchmark = ShekelHPOBenchmark()
