import numpy as np
from deephyper.skopt.moo import pareto_front, hypervolume


class PerformanceEvaluator:
    """ A class defining performance evaluators for JAHS Bench 201 problems.

    Contains the following public methods:

     * `__init__()` constructs a new instance by reading the problem
       definition from environment variables,
     * `hypervolume(pts)` calculates the total hypervolume dominated by
       the current point set, w.r.t. the reference point and filtering out
       solutions that do not dominate the reference point,
     * `refPt()` calculates a reference point for the current problem, and
     * `numPts(pts)` calculates the number of solution points that dominate
       the reference point.

    """

    def __init__(self):
        """ Read the current JAHS-Bench-201 problem definition and initialize. """

        import os

        multiobj = int(os.environ.get("DEEPHYPER_BENCHMARK_MOO", 1))
        prob_name = os.environ.get("DEEPHYPER_BENCHMARK_JAHS_PROB", "fashion_mnist")
        self.p_name = prob_name
        if multiobj:
            self.nobjs = 3
        else:
            self.nobjs = 1

    def hypervolume(self, pts):
        """ Calculate the hypervolume dominated, w.r.t. the reference point.

        Args:
            pts (numpy.ndarray): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            float: The total hypervolume dominated by the current solution
            w.r.t. the reference point, after filtering out points worse than
            the reference point.

        """

        if self.nobjs < 2:
            raise ValueError("Cannot calculate hypervolume for 1 objective")
        if pts.size > 0 and pts[0, 0] > 0:
            filtered_pts = -pts.copy()
        else:
            filtered_pts = pts.copy()
        rp = self.refPt()
        for i in range(pts.shape[0]):
            if np.any(filtered_pts[i, :] > rp):
                filtered_pts[i, :] = rp
        return hypervolume(filtered_pts, rp)

    def refPt(self):
        """ Calculate the reference point for the given problem definition. """

        if self.p_name in ["fashion_mnist"]:
            rp = np.ones(self.nobjs)
            rp[0] = -95
            if self.nobjs > 1:
                rp[1] = 1.75
                rp[2] = 0.6
            return rp
        elif self.p_name in ["cifar10"]:
            rp = np.ones(self.nobjs)
            rp[0] = -90
            if self.nobjs > 1:
                rp[1] = 4.0
                rp[2] = 0.0
            return rp
        elif self.p_name in ["colorectal_histology"]:
            rp = np.ones(self.nobjs)
            rp[0] = -93
            if self.nobjs > 1:
                rp[1] = 4.0
                rp[2] = 0.4
            return rp
        else:
            raise ValueError(f"{self.p_name} is not a valid problem")

    def numPts(self, pts):
        """ Calculate the number of solutions that dominate the reference point.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            int: The number of fi in pts such that all(fi < self.refPt).

        """

        if np.any(pts < 0):
            pareto_pts = pareto_front(-pts)
        else:
            pareto_pts = pareto_front(pts)
        return sum([all(fi <= self.refPt()) for fi in pareto_pts])


if __name__ == "__main__":
    """ Driver code to test performance metrics. """

    result = np.array([[90, -1, -2],
                       [94, -0.1, -0.2],
                       [96, -0.75, -0.1],
                       [99.0, -0.5, -200.0]])

    evaluator = PerformanceEvaluator()

    assert abs(evaluator.hypervolume(result) - 0.5) < 1.0e-8
    assert evaluator.numPts(result) == 1
    assert np.all(np.abs(evaluator.refPt() - np.array([-95.0, 1.75, 0.6]))
                  < 1.0e-8)
