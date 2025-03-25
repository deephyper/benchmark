"""Module for Schwefel benchmark.

Description of the function: https://www.sfu.ca/~ssurjano/schwef.html
"""

import functools
import numpy as np
from deephyper.hpo import HpProblem
from deephyper_benchmark import HPOBenchmark, HPOScorer
from .utils import run_function


def schwefel(x):
    """Schwefel benchmark function."""
    n = len(x)
    y = 418.9829 * n - sum(x * np.sin(np.sqrt(np.abs(x))))
    return -y


class SchwefelScorer(HPOScorer):
    """Define performance evaluators for the Schwefel problem."""

    def __init__(self, nparams=5):
        self.nparams = nparams
        self.x_max = np.full(self.nparams, fill_value=420.9687)
        self.y_max = 0.0


class SchwefelBenchmark(HPOBenchmark):
    """Schwefel benchmark."""

    def __init__(self, nparams: int = 5):
        """Create a Schwefel benchmark."""
        self.nparams = nparams

    @property
    def problem(self):
        """Define the hyperparameter problem."""
        domain = (-500.0, 500.0)
        problem = HpProblem()
        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):
        """Provide the run function for the hyperparameter benchmark."""
        return functools.partial(run_function, bb_func=schwefel)

    @property
    def scorer(self):
        """Provide the scorer for the hyperparameter benchmark."""
        return SchwefelScorer(self.nparams)
