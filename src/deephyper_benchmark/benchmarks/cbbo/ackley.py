"""here."""

import functools

import numpy as np
from deephyper.hpo import HpProblem

from deephyper_benchmark import HPOBenchmark, HPOScorer

from .utils import run_function


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
    return -y


class AckleyScorer(HPOScorer):
    """A class defining performance evaluators for the Ackley problem."""

    def __init__(
        self,
        nparams,
        nslack,
        offset=0.0,
    ):
        self.nparams = nparams
        self.nslack = nslack
        self.offset = offset
        self.x_max = np.asarray([offset for _ in range(self.nparams)])
        self.y_max = 0.0


class AckleyBenchmark(HPOBenchmark):
    """Ackley benchmark.

    Args:
        nparams (int, optional): the number of parameters in the problem.
        offset (int, optional): the offset in the space of parameters.
        nslack (int, optional): the number of additional slack parameters in the problem.
    """

    def __init__(self, nparams: int = 5, offset: int = -4.0, nslack: int = 0) -> None:
        self.nparams = nparams
        self.nslack = nslack
        assert offset <= 32.768 and offset >= -32.768, (
            "offset must be in [-32.768, 32.768] to keep the same maximum value."
        )
        self.offset = offset

    @property
    def problem(self):  # noqa: D102
        # The original range is simetric (-32.768, 32.768) but we make it less simetric to avoid
        # Grid sampling or QMC sampling to directly hit the optimum...
        domain = (-32.768 + self.offset, 32.768 + self.offset)
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
        return functools.partial(run_function, bb_func=ackley)

    @property
    def scorer(self):  # noqa: D102
        return AckleyScorer(self.nparams, self.nslack, self.offset)
