"""Module defining the problem and run-function of the benchmark."""

import os
import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob
from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


def easom(x):
    assert len(x) == 2
    y = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
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

    return -easom(x)


class EasomHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Ackley problem."""

    def __init__(self, offset=0):
        self.x_max = np.array([np.pi, np.pi])
        self.y_max = 1.0

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


class EasomHPOBenchmark(HPOBenchmark):
    def refresh_settings(self):
        self.DEEPHYPER_BENCHMARK_OFFSET = float(os.environ.get("DEEPHYPER_BENCHMARK_OFFSET", 4.0))

    @property
    def problem(self):
        # Grid sampling or QMC sampling to directly hit the optimum...
        domain = (
            -100.0 - self.DEEPHYPER_BENCHMARK_OFFSET,
            100.0 - self.DEEPHYPER_BENCHMARK_OFFSET,
        )
        problem = HpProblem()
        for i in range(2):
            problem.add_hyperparameter(domain, f"x{i}")

        return problem

    @property
    def run_function(self):
        return run_function

    @property
    def scorer(self):
        return EasomHPOScorer(self.DEEPHYPER_BENCHMARK_OFFSET)


benchmark = EasomHPOBenchmark()
