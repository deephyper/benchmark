import os

import numpy as np


class PerformanceEvaluator:
    """A class defining performance evaluators for the DTLZ problems.s"""

    def __init__(self):
        """Read the current problem defn from environment vars."""

        self.p_num = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))
        self.x_min = np.zeros(self.p_num)
        self.y_min = 0.0

    def simple_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the regret of a list of given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        return y - self.y_min

    def cumul_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the cumulative regret of an array of ordered given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of cumulative regret values.
        """
        return np.cumsum(self.simple_regret(y))
