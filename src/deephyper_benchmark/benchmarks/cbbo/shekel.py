"""Module for Shekel benchmark."""

import functools
import numpy as np
from deephyper.hpo import HpProblem
from deephyper_benchmark import HPOBenchmark, HPOScorer
from .utils import run_function


def shekel(x):
    """Shekel benchmark function."""
    m = 10
    beta = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]).T

    C = np.array(
        [
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        ]
    )

    y = -sum([1 / (np.sum((x - C[:, i]) ** 2) + beta[i]) for i in range(m)])

    return y


class ShekelScorer(HPOScorer):
    """Define performance evaluators for the Shekel problem."""

    def __init__(self, nparams=4):
        self.nparams = nparams


class ShekelBenchmark(HPOBenchmark):
    """Shekel benchmark."""

    def __init__(self, nparams=4):
        """Create a Shekel benchmark."""
        self.nparams = nparams

    @property
    def problem(self):
        """Define the hyperparameter problem."""
        domain = (0.0, 10.0)
        problem = HpProblem()

        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")

    @property
    def run_function(self):
        """Provide the run function for the hyperparameter benchmark."""
        return functools.partial(run_function, bb_func=shekel)

    @property
    def scorer(self):
        """Provide the scorer for the hyperparameter benchmark."""
        return ShekelScorer(self.nparams)
