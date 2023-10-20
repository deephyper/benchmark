import os
import warnings

import numpy as np


class PerformanceEvaluator:
    """A class defining performance evaluators for the Ackley problem."""

    def __init__(self):
        """Read the current problem defn from environment vars."""

        self.p_num = int(os.environ.get("DEEPHYPER_BENCHMARK_NDIMS", 5))
        self.x_min = None
        self.y_min = None

        if self.p_num == 2:
            self.x_min = np.array([2.20, 1.57])
            self.y_min = -1.8013
        elif self.p_num == 5:
            self.y_min = -4.687658
        elif self.p_num == 10:
            self.y_min = -9.66015

        if self.x_min is None:
            warnings.warn(
                f"PerformanceEvaluator: unknown optimum location 'x' for problem dimension {self.p_num}"
            )

        if self.y_min is None:
            warnings.warn(
                f"PerformanceEvaluator: unknown optimum value 'y' for problem dimension {self.p_num}"
            )

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
