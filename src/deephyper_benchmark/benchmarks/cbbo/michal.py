"""Module for Michal benchmark.

Description of the function: https://www.sfu.ca/~ssurjano/michal.html
"""

import functools

import numpy as np
from deephyper.hpo import HpProblem

from deephyper_benchmark import HPOBenchmark, HPOScorer

from .utils import run_function


def michal(x, m=10):
    """Michalewicz function benchmark."""
    ix2 = np.arange(1, len(x) + 1) * x**2
    y = -np.sum(np.sin(x) * np.power(np.sin(ix2 / np.pi), 2 * m))
    return -y


class MichalScorer(HPOScorer):
    """Define performance evaluators for the Michal problem."""

    def __init__(self, nparams=2):
        assert nparams in [2, 5, 10], (
            "nparams should be in [2, 5, 10] otherwise the solution is unknown."
        )
        self.nparams = nparams
        if self.nparams == 2:
            self.x_max = np.array([2.20, 1.57])
            self.y_max = 1.8013
        elif self.nparams == 5:
            self.x_max = None
            self.y_max = 4.687658
        elif self.nparams == 10:
            self.x_max = None
            self.y_max = 9.66015


class MichalBenchmark(HPOBenchmark):
    """Michal benchmark."""

    def __init__(self, nparams: int = 5):
        """Create a Michal benchmark."""
        self.nparams = nparams

    @property
    def problem(self):
        """Define the hyperparameter problem."""
        domain = (0.0, np.pi)
        problem = HpProblem()
        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):
        """Provide the run function for the hyperparameter benchmark."""
        return functools.partial(run_function, bb_func=michal)

    @property
    def scorer(self):
        """Provide the scorer for the hyperparameter benchmark."""
        return MichalScorer(self.nparams)
