"""here."""

import functools

import numpy as np
from deephyper.hpo import HpProblem

from deephyper_benchmark import HPOBenchmark, HPOScorer

from .utils import run_function


def easom(x):
    """Easom function.

    Description of the function: https://www.sfu.ca/~ssurjano/easom.html
    """
    assert len(x) == 2
    y = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
    return -y


class EasomScorer(HPOScorer):
    """A class defining performance evaluators for the Easom problem."""

    def __init__(self):
        self.nparams = 2
        self.x_max = np.array([np.pi, np.pi])
        self.y_max = 1.0


class EasomBenchmark(HPOBenchmark):
    """Easom benchmark."""

    def __init__(self) -> None:
        self.nparams = 2

    @property
    def problem(self):  # noqa: D102
        domain = (-100.0, 100.0)
        problem = HpProblem()
        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):  # noqa: D102
        return functools.partial(run_function, bb_func=easom)

    @property
    def scorer(self):  # noqa: D102
        return EasomScorer()
