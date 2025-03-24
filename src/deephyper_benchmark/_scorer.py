import abc

import numpy as np
from deephyper.skopt.moo import hypervolume
from pydantic import BaseModel


class Scorer(BaseModel, abc.ABC):
    pass


class HPOScorer(Scorer):

    y_max: float
    def simple_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the regret of a list of given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        return self.y_max - y

    def cumul_regret(self, y: np.ndarray) -> np.ndarray:
        """Compute the cumulative regret of an array of ordered given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of cumulative regret values.
        """
        return np.cumsum(self.simple_regret(y))


class MultiObjHPOScorer(Scorer):
    nobj: int

    def hypervolume_score(self, pts: np.ndarray):
        """Calculate the hypervolume dominated by soln, wrt the Nadir point.

        Args:
            pts (np.ndarray): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            float: The total hypervolume dominated by the current solution,
            filtering out points worse than the Nadir point and using the
            Nadir point as the reference.

        """
        if np.any(pts < 0):
            filtered_pts = -pts.copy()
        else:
            filtered_pts = pts.copy()
        nadir = self.nadir_point
        for i in range(pts.shape[0]):
            if np.any(filtered_pts[i, :] > nadir):
                filtered_pts[i, :] = nadir
        return hypervolume(filtered_pts, nadir)

    @property
    @abc.abstractmethod
    def nadir_point(self):
        pass

    def hypervolume(self, y: np.ndarray) -> np.ndarray:
        """Compute the regret of a list of given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        scores = np.zeros(y.shape[0])
        for i in range(1, y.shape[0] + 1):
            scores[i - 1] = self.hypervolume_score(y[:i])
        return scores
