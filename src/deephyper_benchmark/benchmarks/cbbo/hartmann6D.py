"""here."""

import functools

import numpy as np
from deephyper.hpo import HpProblem

from deephyper_benchmark import HPOBenchmark, HPOScorer

from .utils import run_function


def hartmann6D(x):  # noqa: D103
    """Hartmann6D function benchmark.

    Description of the function: https://www.sfu.ca/~ssurjano/hart6.html
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    X = np.array([x for _ in range(4)])
    inner = np.sum(np.multiply(A, np.square(X - P)), axis=1)
    outer = np.sum(alpha * np.exp(-inner))
    y = -(2.58 + outer) / 1.94
    return -y


class Hartmann6DScorer(HPOScorer):
    """A class defining performance evaluators for the Hartmann6D problem."""

    def __init__(self):
        self.nparams = 6
        self.x_max = np.asarray(
            [
                0.20169,
                0.150011,
                0.476874,
                0.275332,
                0.311652,
                0.6573,
            ]
        )
        self.y_max = 3.32237


class Hartmann6DBenchmark(HPOBenchmark):
    """Hartmann6D benchmark."""

    def __init__(self) -> None:
        self.nparams = 6

    @property
    def problem(self):  # noqa: D102
        domain = (0.0, 1.0)
        problem = HpProblem()
        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):  # noqa: D102
        return functools.partial(run_function, bb_func=hartmann6D)

    @property
    def scorer(self):  # noqa: D102
        return Hartmann6DScorer()
