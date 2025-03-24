"""Module for Michal benchmark."""

import functools
import numpy as np
from deephyper.hpo import HpProblem
from deephyper_benchmark import HPOBenchmark, HPOScorer


def michal(x, m=10):
    """Michal benchmark function."""
    ix2 = np.arange(1, len(x) + 1) * x**2
    y = -np.sum(np.sin(x) * np.power(np.sin(ix2 / np.pi), 2 * m))
    return y


class MichalScorer(HPOScorer):
    """Define performance evaluators for the Michal problem."""

    def __init__(self, nparams=2):
        self.nparams = nparams


class MichalBenchmark(HPOBenchmark):
    """Michal benchmark."""

    def __init__(self, nparams=2):
        """Create a Michal benchmark."""
        self.nparams = nparams

    @property
    def problem(self):
        """Define the hyperparameter problem."""
        domain = (0, np.pi)
        problem = HpProblem()

        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")

    @property
    def run_function(self):
        """Provide the run function for the hyperparameter benchmark."""
        return functools.partial(run_function, bb_func=michal)

    @property
    def scorer(self):
        """Provide the scorer for the hyperparameter benchmark."""
        return MichalScorer(self.nparams)
