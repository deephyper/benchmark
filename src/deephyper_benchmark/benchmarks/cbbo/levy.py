"""Module for Levy benchmark."""

import functools
import numpy as np
from deephyper.hpo import HpProblem
from deephyper_benchmark import HPOBenchmark, HPOScorer
from .utils import run_function


def levy(x):
    """Levy benchmark function."""
    z = 1 + (x - 1) / 4
    func = (
        np.sin(np.pi * z[0]) ** 2
        + np.sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
        + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2)
    )
    return func


class LevyScorer(HPOScorer):
    """A class defining performance evaluators for the Levy problem."""

    def __init__(self, nparams: int = 5):
        self.nparams = nparams


class LevyBenchmark(HPOBenchmark):
    """Levy benchmark."""

    def __init__(self, nparams=5):
        """Create a Levy benchmark."""
        self.nparams = nparams

    @property
    def problem(self):
        """Define the hyperparameter problem."""
        domain = (-10.0, 10.0)
        problem = HpProblem()

        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")

    @property
    def run_function(self):
        """Provide the run function for the hyperparameter benchmark."""
        return functools.partial(run_function, bb_func=levy)

    @property
    def scorer(self):
        """Provide the scorer for the hyperparameter benchmark."""
        return LevyScorer(self.nparams)
