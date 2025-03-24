"""here."""

import functools

import numpy as np
from deephyper.hpo import HpProblem

from deephyper_benchmark import HPOBenchmark, HPOScorer

from .utils import run_function


def griewank(x, fr=4000):  # noqa: D103
    n = len(x)
    j = np.arange(1.0, n + 1)
    s = np.sum(x**2)
    p = np.prod(np.cos(x / np.sqrt(j)))
    y = s / fr - p + 1
    return -y


class GriewankScorer(HPOScorer):
    """A class defining performance evaluators for the Griewank problem."""

    def __init__(
        self,
        nparams,
        nslack,
        offset=0.0,
    ):
        self.nparams = nparams
        self.nslack = nslack
        self.offset = offset
        self.x_max = np.full(self.nparams, fill_value=offset)
        self.x_max[nparams - nslack :] = np.nan
        self.y_max = 0.0


class GriewankBenchmark(HPOBenchmark):
    """Griewank benchmark.

    Args:
        nparams (int, optional): the number of parameters in the problem.
        offset (int, optional): the offset in the space of parameters.
        nslack (int, optional): the number of additional slack parameters in the problem.
    """

    def __init__(self, nparams: int = 5, offset: int = -4.0, nslack: int = 0) -> None:
        self.nparams = nparams
        assert offset <= 600.0 and offset >= -600.0, (
            "offset must be in [-600.0, 600.0] to keep the same maximum value."
        )
        self.offset = offset
        self.nslack = nslack

    @property
    def problem(self):  # noqa: D102
        domain = (-600.0 + self.offset, 600.0 + self.offset)
        problem = HpProblem()
        for i in range(self.nparams - self.nslack):
            problem.add_hyperparameter(domain, f"x{i}")

        for i in range(
            self.nparams - self.nslack,
            self.nparams,
        ):
            problem.add_hyperparameter(domain, f"z{i}")
        return problem

    @property
    def run_function(self):  # noqa: D102
        return functools.partial(run_function, bb_func=griewank)

    @property
    def scorer(self):  # noqa: D102
        return GriewankScorer(self.nparams, self.nslack, self.offset)
