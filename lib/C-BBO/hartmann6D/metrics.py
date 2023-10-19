import os

import numpy as np


class PerformanceEvaluator:
    """A class defining performance evaluators for the Hartmann6D problem."""

    def __init__(self):
        """Read the current problem defn from environment vars."""

        self.p_num = 6
        self.x_min = np.array(
            [
                0.20169,
                0.150011,
                0.476874,
                0.275332,
                0.311652,
                0.6573,
            ]
        )
        self.y_min = -3.32237

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
