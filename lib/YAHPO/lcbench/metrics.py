import json
import os

import numpy as np

from .hpo import bench as BENCHMARK


class PerformanceEvaluator:
    """A class defining performance evaluators for the YAHPO/lcbench problems."""

    def __init__(self):
        """Read the current problem defn from environment vars."""

        df = BENCHMARK.target_stats

        self.x_min = None

        self.y_metric = "cross_entropy"

        cond = (df["metric"] == "val_cross_entropy") & (df["statistic"] == "min")
        self.y_min_valid = df[cond].iloc[0]["value"]

        cond = (df["metric"] == "test_cross_entropy") & (df["statistic"] == "min")
        self.y_min_test = df[cond].iloc[0]["value"]

    def simple_regret_valid(self, y_valid: np.ndarray) -> np.ndarray:
        """Compute the regret of the objective (validation RMSE) of a list of ordered solutions.

        Args:
            y_valid (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        return y_valid - self.y_min_valid

    def cumul_regret_valid(self, y_valid: np.ndarray) -> np.ndarray:
        """Compute the cumulative regret of the objective (validation RMSE) aon n array of ordered solutions.

        Args:
            y_valid (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of cumulative regret values.
        """
        return np.cumsum(self.simple_regret_valid(y_valid))

    def simple_regret_test(self, y_test: np.ndarray) -> np.ndarray:
        """Compute the regret of the test RMSE of a list of ordered solutions.

        Args:
            y_test (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        return y_test - self.y_min_test

    def cumul_regret_test(self, y_test: np.ndarray) -> np.ndarray:
        """Compute the cumulative regret of the test RMSE aon n array of ordered solutions.

        Args:
            y_test (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of cumulative regret values.
        """
        return np.cumsum(self.simple_regret_test(y_test))
