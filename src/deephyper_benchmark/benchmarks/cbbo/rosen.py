"""Module for Rosen benchmark."""

import functools
from deephyper.hpo import HpProblem
from deephyper_benchmark import HPOBenchmark, HPOScorer
from scipy.optimize import rosen
from .utils import run_function


class RosenScorer(HPOScorer):
    """Define performance evaluators for the Rosen problem."""

    def __init__(self, nparams=5):
        self.nparams = nparams


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

    @property
    def run_function(self):
        """Provide the run function for the hyperparameter benchmark."""
        return functools.partial(run_function, bb_func=rosen)

    @property
    def scorer(self):
        """Provide the scorer for the hyperparameter benchmark."""
        return RosenScorer(self.nparams)
