import os
import numpy as np
from deephyper.skopt.moo import pareto_front, hypervolume


class PerformanceEvaluator:
    """ A class defining performance evaluators for JAHS Bench 201 problems.

    Contains the following public methods:

     * `__init__()` constructs a new instance by reading the problem defn
       from environment variables,
     * `hypervolume(pts)` calculates the total hypervolume dominated by
       the current solution, using the Nadir point as the reference point
       and filtering out solutions that do not dominate the Nadir point,
     * `nadirPt()` calculates the Nadir point for the current problem,
     * `numPts(pts)` calculates the number of solution points that dominate
       the Nadir point, and

    """

    def __init__(self, p_name="fashion_mnist"):
        """ Read the current DTLZ problem defn from environment vars. """

        self.p_name = p_name
        multiobj = int(os.environ.get("DEEPHYPER_BENCHMARK_MOO", 1))
        if multiobj:
            self.nobjs = 3
        else:
            self.nobjs = 1

    def hypervolume(self, pts):
        """ Calculate the hypervolume dominated by soln, wrt the Nadir point.

        Args:
            pts (numpy.ndarray): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            float: The total hypervolume dominated by the current solution,
            filtering out points worse than the Nadir point and using the
            Nadir point as the reference.

        """

        if self.nobjs < 2:
            raise ValueError("Cannot calculate hypervolume for 1 objective")
        if pts.size > 0 and pts[0, 0] > 0:
            filtered_pts = -pts.copy()
        else:
            filtered_pts = pts.copy()
        nadir = self.nadirPt()
        for i in range(pts.shape[0]):
            if np.any(filtered_pts[i, :] > nadir):
                filtered_pts[i, :] = nadir
        return hypervolume(filtered_pts, nadir)

    def nadirPt(self):
        """ Calculate the Nadir point for the given problem definition. """

        if self.p_name in ["fashion_mnist"]:
            nadir = np.ones(self.nobjs)
            nadir[0] = 88
            if self.nobjs > 1:
                nadir[1] = 10.0
                nadir[2] = 100.0
            return nadir
        elif self.p_name in ["cifar10"]:
            nadir = np.ones(self.nobjs)
            nadir[0] = 50
            if self.nobjs > 1:
                nadir[1] = 10.0
                nadir[2] = 100.0
            return nadir
        elif self.p_name in ["colorectal_history"]:
            nadir = np.ones(self.nobjs)
            nadir[0] = 81
            if self.nobjs > 1:
                nadir[1] = 10.0
                nadir[2] = 100.0
            return nadir
        else:
            raise ValueError(f"{self.p_name} is not a valid problem")

    def numPts(self, pts):
        """ Calculate the number of solutions that dominate the Nadir point.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            int: The number of fi in pts such that all(fi < self.nadirPt).

        """

        if np.any(pts < 0):
            pareto_pts = pareto_front(-pts)
        else:
            pareto_pts = pareto_front(pts)
        return sum([all(fi <= self.nadirPt()) for fi in pareto_pts])


if __name__ == "__main__":
    """ Driver code to test performance metrics. """

    result = np.array([[80, -8, -10], [90, -9, -90], [10, -9.1, -99], [99.0, -1.0, -200.0]])

    evaluator = PerformanceEvaluator()

    assert abs(evaluator.hypervolume(result) - 14500) < 1.0e-8
    assert evaluator.numPts(result) == 2
    assert np.all(np.abs(evaluator.nadirPt() - np.array([0, 10, 100]))
                  < 1.0e-8)
