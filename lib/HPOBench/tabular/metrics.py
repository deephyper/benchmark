import json
import os

import numpy as np

from .hpo import b as BENCHMARK


class PerformanceEvaluator:
    """A class defining performance evaluators for the DTLZ problems."""

    def __init__(self):
        """Read the current problem defn from environment vars."""

        # Retrieve the Best Configuration
        configs, te, ve = [], [], []
        for k in BENCHMARK.benchmark.data.keys():
            configs.append(json.loads(k))
            te.append(np.min(BENCHMARK.benchmark.data[k]["final_test_error"]))
            ve.append(np.min(np.mean(BENCHMARK.benchmark.data[k]["valid_mse"], axis=0)))

        idx_opt_ve = np.argmin(ve)
        idx_opt_te = np.argmin(te)

        # Configuration, Validation Error, Test Error based on the best validation error
        self.x_min_on_valid = ve[idx_opt_ve]
        self.y_min_valid_on_valid = ve[idx_opt_ve]
        self.y_min_test_on_valid = te[idx_opt_ve]

        # Configuration, Validation Error, Test Error based on the best final test error
        # only available for this benchmark
        self.x_min_on_test = dict(configs[idx_opt_te])
        self.y_min_valid_on_test = ve[idx_opt_te]
        self.y_min_test_on_test = te[idx_opt_te]

    def simple_regret_valid(self, y_valid: np.ndarray) -> np.ndarray:
        """Compute the regret of the objective (validation RMSE) of a list of ordered solutions.

        Args:
            y_valid (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        return y_valid - self.y_min_valid_on_valid

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
        return y_test - self.y_min_test_on_test

    def cumul_regret_test(self, y_test: np.ndarray) -> np.ndarray:
        """Compute the cumulative regret of the test RMSE aon n array of ordered solutions.

        Args:
            y_test (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of cumulative regret values.
        """
        return np.cumsum(self.simple_regret_test(y_test))
