"""here."""

import numpy as np
import time

from deephyper.evaluator import profile
from deephyper.evaluator import RunningJob
from deephyper.hpo import HpProblem
from deephyper_benchmark import HPOBenchmark, HPOScorer


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """Ackley function.

    Description of the function: https://www.sfu.ca/~ssurjano/ackley.html
    """
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    y = term1 + term2 + a + np.exp(1)
    return y


@profile
def run_function(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:  # noqa: D103
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return -ackley(x)


class AckleyScorer(HPOScorer):
    """A class defining performance evaluators for the Ackley problem."""

    def __init__(
        self,
        nparams,
        nslack,
    ):
        self.nparams = nparams
        self.x_max = np.full(self.nparams, fill_value=0.0)
        self.x_max[nparams - nslack :] = np.nan
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


class AckleyBenchmark(HPOBenchmark):
    """Ackley benchmark.

    Args:
        nparams (int, optional): the number of parameters in the problem.
        offset (int, optional): the offset in the space of parameters.
        nslack (int, optional): the number of additional slack parameters in the problem.
    """

    def __init__(self, nparams: int = 5, offset: int = -4.0, nslack: int = 0) -> None:
        self.nparams = nparams
        assert offset <= 32.768 and offset >= -32.768, (
            "offset must be in [-32.768, 32.768] to keep the same maximum value."
        )
        self.offset = offset
        self.nslack = nslack

    @property
    def problem(self):  # noqa: D102
        # The original range is simetric (-32.768, 32.768) but we make it less simetric to avoid
        # Grid sampling or QMC sampling to directly hit the optimum...
        domain = (
            -32.768 + self.offset,
            32.768 + self.offset,
        )
        problem = HpProblem()
        for i in range(self.nparams - self.nslack):
            problem.add_hyperparameter(domain, f"x{i}")

        # Add slack/dummy dimensions (useful to test predicors which are sensitive
        # to unimportant features)
        for i in range(
            self.nparams - self.nslack,
            self.nparams,
        ):
            problem.add_hyperparameter(domain, f"z{i}")
        return problem

    @property
    def run_function(self):  # noqa: D102
        return run_function

    @property
    def scorer(self):  # noqa: D102
        return AckleyScorer(self.nparams, self.nslack)
