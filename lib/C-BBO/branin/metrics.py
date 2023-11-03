import os

import numpy as np


class PerformanceEvaluator:
    """A class defining performance evaluators for the Ackley problem."""

    def __init__(self):

        self.p_num = 2
        self.x_min = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.y_min = 0.397887

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
