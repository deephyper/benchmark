import os
import numpy as np
from deephyper.skopt.moo import pareto_front, hypervolume


class PerformanceEvaluator:
    """ A class defining performance evaluators for the DTLZ problems.

    Contains the following public methods:

     * `__init__()` constructs a new instance by reading the problem defn
       from environment variables,
     * `hypervolume(pts)` calculates the total hypervolume dominated by
       the current solution, using the Nadir point as the reference point
       and filtering out solutions that do not dominate the Nadir point,
     * `nadirPt()` calculates the Nadir point for the current problem,
     * `numPts(pts)` calculates the number of solution points that dominate
       the Nadir point, and
     * `rmse(pts)` calculates the RMSE where the error in each point is
       approximated by the 2-norm distance to the nearest solution point.

    """

    def __init__(self):
        """ Read the current DTLZ problem defn from environment vars. """

        self.p_num = os.environ.get("DEEPHYPER_BENCHMARK_DTLZ_PROB", "2")
        self.nobjs = int(os.environ.get("DEEPHYPER_BENCHMARK_NOBJS", 2))

    def hypervolume(self, pts):
        """ Calculate the hypervolume dominated by soln, wrt the Nadir point.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            float: The total hypervolume dominated by the current solution,
            filtering out points worse than the Nadir point and using the
            Nadir point as the reference.

        """

        filtered_pts = pts.copy()
        nadir = self.nadirPt()
        for i in range(pts.shape[0]):
            if np.any(filtered_pts[i, :] > nadir):
                filtered_pts[i, :] = nadir
        return hypervolume(filtered_pts, nadir)

    def nadirPt(self):
        """ Calculate the Nadir point for the given problem definition. """

        if self.p_num == "1":
            return np.ones(self.nobjs) * 0.5
        elif self.p_num in ["2", "3", "4", "5", "6"]:
            return np.ones(self.nobjs)
        elif self.p_num == "7":
            nadir = np.ones(self.nobjs)
            nadir[self.nobjs - 1] = self.nobjs * 2.0
            return nadir
        else:
            raise ValueError(f"DTLZ{self.p_num} is not a valid problem")

    def numPts(self, pts):
        """ Calculate the number of solutions that dominate the Nadir point.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            int: The number of fi in pts such that all(fi < self.nadirPt).

        """

        pareto_pts = pareto_front(pts)
        return sum([all(fi <= self.nadirPt()) for fi in pareto_pts])

    def rmse(self, pts):
        """ Calculate the RMSE for a set of objective points.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            float: The RMSE over all points in pts.

        """

        pareto_pts = pareto_front(pts)
        if self.p_num == "1":
            dists = self._dtlz1Dist(pareto_pts)
        elif self.p_num in ["2", "3", "4", "5", "6"]:
            dists = self._dtlz2Dist(pareto_pts)
        elif self.p_num == "7":
            dists = self._dtlz7Dist(pareto_pts)
        else:
            raise ValueError(f"DTLZ{self.p_num} is not a valid problem")
        return np.sqrt(np.sum(dists ** 2) / len(dists))

    def _dtlz1Dist(self, pts):
        """ Calculate the distance from each fi to the nearest soln in DTLZ1.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            numpy.ndarray: A 1d array of distances to the nearest solution
            point for DTLZ1.
            

        """

        return np.array([np.linalg.norm(0.5 * fi / np.sum(fi) - fi)
                         for fi in pts])

    def _dtlz2Dist(self, pts):
        """ Calculate the distance from each fi to the unit sphere.

        Note: Works for DTLZ2-6

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            numpy.ndarray: A 1d array of distances to the surface of the
            unit sphere.

        """

        return np.array([np.linalg.norm(fi / np.linalg.norm(fi) - fi)
                         for fi in pts])

    def _dtlz7Dist(self, pts):
        """ Calculate the distance from each fi to the nearest soln in DTLZ7.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            numpy.ndarray: A 1d array of distances to the nearest solution
            point to DTLZ7.

        """

        # Project each point onto DTLZ7 solution and calculate difference
        pts_proj = []
        for fi in pts:
            gx = 1.0
            hx = float(self.nobjs)
            for j in range(self.nobjs-1):
                hx = hx - ((fi[j] / (1.0 + gx)) * (1.0 + np.sin(3.0 * np.pi
                                                                * fi[j])))
            pts_proj.append((1.0 + gx) * hx)
        return np.array([np.abs(fi[-1] - fj) for fi, fj in zip(pts, pts_proj)])


if __name__ == "__main__":
    """ Driver code to test performance metrics. """

    os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "1" # DTLZ1 problem
    dtlz1_eval = PerformanceEvaluator()
    s1 = np.array([[0.5, 0], [0, 0.5], [.25, .25], [0.2, 0.8]])
    os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "2" # DTLZ2 problem
    dtlz2_eval = PerformanceEvaluator()
    s2 = np.array([[1, 0], [0, 1], [1/np.sqrt(2), 1/np.sqrt(2)], [0.25, 2]])
    os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "7" # DTLZ7 problem
    dtlz7_eval = PerformanceEvaluator()
    s7 = np.array([[0, 4], [1, 3], [.5, 4], [0.5, 6]])

    assert abs(dtlz1_eval.hypervolume(s1) - .0625) < 1.0e-8
    assert np.all(np.abs(dtlz1_eval.nadirPt() - 0.5) < 1.0e-8)
    assert dtlz1_eval.numPts(s1) == 3
    assert abs(dtlz1_eval.rmse(s1)) < 1.0e-8

    assert abs(dtlz2_eval.hypervolume(s2) - (1.5 - np.sqrt(2))) < 1.0e-8
    assert np.all(np.abs(dtlz2_eval.nadirPt() - 1) < 1.0e-8)
    assert dtlz2_eval.numPts(s2) == 3
    assert abs(dtlz2_eval.rmse(s2)) < 1.0e-8

    assert abs(dtlz7_eval.hypervolume(s7)) < 1.0e-8
    assert np.all(np.abs(dtlz7_eval.nadirPt() - np.array([1, 4])) < 1.0e-8)
    assert dtlz7_eval.numPts(s7) == 2
    assert abs(dtlz7_eval.rmse(s7)) < 1.0e-8
