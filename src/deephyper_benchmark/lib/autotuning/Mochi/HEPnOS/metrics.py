import os

import numpy as np

from .hpo import load_data, problem

class PerformanceEvaluator:
    """A class defining performance evaluators for the HEPnOS problem."""

    def __init__(self):
        """Read the current problem defn from environment vars."""

        df = load_data()
        df = df[df["objective"] > 0]
        idx_max = df["objective"].argmin()
        self.x_min = df[problem.hyperparameter_names].iloc[idx_max].to_dict()
        self.y_min = df.iloc[idx_max]["objective"]

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
