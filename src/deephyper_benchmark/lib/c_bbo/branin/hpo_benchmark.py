"""Module defining the problem and run-function of the benchmark."""

import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob
from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


def branin(x):
    assert len(x) == 2
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
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

    return -branin(x)


class BraninHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Ackley problem."""

    def __init__(self):
        self.p_num = 2
        self.x_max = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.y_max = -0.397887

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


class BraninHPOBenchmark(HPOBenchmark):
    @property
    def problem(self):
        problem = HpProblem()
        problem.add_hyperparameter((-5.0, 10.0), "x0")
        problem.add_hyperparameter((0.0, 15.0), "x1")
        return problem

    @property
    def run_function(self):
        return run_function

    @property
    def scorer(self):
        return BraninHPOScorer()


benchmark = BraninHPOBenchmark()
