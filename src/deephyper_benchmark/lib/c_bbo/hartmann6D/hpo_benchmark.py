"""Module defining the problem and run-function of the benchmark."""

import time

import numpy as np

from deephyper.hpo import HpProblem
from deephyper.evaluator import profile, RunningJob
from deephyper_benchmark import HPOBenchmark, HPOScorer

__all__ = ["benchmark"]


def hartmann6D(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    X = np.array([x for _ in range(4)])
    inner = np.sum(np.multiply(A, np.square(X - P)), axis=1)
    outer = np.sum(alpha * np.exp(-inner))
    y = -(2.58 + outer) / 1.94
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

    return -hartmann6D(x)


class Hartmann6DHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Hartmann6D problem."""

    def __init__(self):
        self.p_num = 6
        self.x_max = np.array(
            [
                0.20169,
                0.150011,
                0.476874,
                0.275332,
                0.311652,
                0.6573,
            ]
        )
        self.y_max = 3.32237

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


class Hartmann6DHPOBenchmark(HPOBenchmark):
    @property
    def problem(self):
        # The original range is simetric (-32.768, 32.768) but we make it less simetric to avoid
        # Grid sampling or QMC sampling to directly hit the optimum...
        domain = (0.0, 1.0)
        problem = HpProblem()
        for i in range(6):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):
        return run_function

    @property
    def scorer(self):
        return Hartmann6DHPOScorer(
            self.DEEPHYPER_BENCHMARK_NDIMS,
            self.DEEPHYPER_BENCHMARK_NDIMS_SLACK,
            self.DEEPHYPER_BENCHMARK_OFFSET,
        )


benchmark = Hartmann6DHPOBenchmark()
