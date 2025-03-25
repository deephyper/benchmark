"""here."""

import functools

import numpy as np
from deephyper.hpo import HpProblem

from deephyper_benchmark import HPOBenchmark, HPOScorer

from .utils import run_function


def branin(x):
    """Branin function.

    Description of the function: https://www.sfu.ca/~ssurjano/branin.html"
    """
    assert len(x) == 2
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
    return -y


class BraninHPOScorer(HPOScorer):
    """A class defining performance evaluators for the Ackley problem."""

    def __init__(self):
        self.nparams = 2
        self.x_max = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.y_max = -0.397887


class BraninBenchmark(HPOBenchmark):
    """Branin benchmark."""

    @property
    def problem(self):  # noqa: D102
        problem = HpProblem()
        problem.add_hyperparameter((-5.0, 10.0), "x0")
        problem.add_hyperparameter((0.0, 15.0), "x1")
        return problem

    @property
    def run_function(self):  # noqa: D102
        return functools.partial(run_function, bb_func=branin)

    @property
    def scorer(self):  # noqa: D102
        return BraninHPOScorer()
