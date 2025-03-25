"""Module for Rosen benchmark.

Description of the function: https://www.sfu.ca/~ssurjano/rosen.html
"""

import functools

import numpy as np
from deephyper.hpo import HpProblem
from scipy.optimize import rosen

from deephyper_benchmark import HPOBenchmark, HPOScorer

from .utils import run_function


def rosen_(x):  # noqa: D103
    return -rosen(x)


class RosenScorer(HPOScorer):
    """Define performance evaluators for the Rosen problem."""

    def __init__(self, nparams: int = 5):
        self.nparams = nparams
        self.x_max = np.ones(self.nparams)
        self.y_max = 0.0


class RosenBenchmark(HPOBenchmark):
    """Rosen benchmark."""

    def __init__(self, nparams=5):
        """Create a Rosen benchmark."""
        self.nparams = nparams

    @property
    def problem(self):
        """Define the hyperparameter problem."""
        domain = (-5.0, 10.0)
        problem = HpProblem()
        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):
        """Provide the run function for the hyperparameter benchmark."""
        return functools.partial(run_function, bb_func=rosen_)

    @property
    def scorer(self):
        """Provide the scorer for the hyperparameter benchmark."""
        return RosenScorer(self.nparams)
