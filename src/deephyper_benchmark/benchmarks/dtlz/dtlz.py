"""here."""

import functools
import time
from typing import Optional

import numpy as np
from deephyper.evaluator import RunningJob, profile
from deephyper.hpo import HpProblem
from deephyper.skopt.moo import hypervolume, pareto_front

from deephyper_benchmark import HPOBenchmark, MultiObjHPOScorer

from . import model as dtlz


@profile
def run_function(  # noqa: D103
    job: RunningJob,
    dtlz_obj,
    nobj: int,
    with_failures=False,
    sleep=False,
    sleep_mean=60,
    sleep_noise=20,
) -> dict:
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    ff = [-fi for fi in dtlz_obj(x)]

    if with_failures:
        if any(xi < 0.25 for xi in x[nobj - 1 :]):
            ff = ["F" for _ in ff]

    return ff


class DTLZScorer(MultiObjHPOScorer):
    """A class defining performance evaluators for the DTLZ problems.

    Contains the following public methods:

     * `hypervolume(pts)` calculates the total hypervolume dominated by
       the current solution, using the Nadir point as the reference point
       and filtering out solutions that do not dominate the Nadir point,
     * `nadirPt()` calculates the Nadir point for the current problem,
     * `numPts(pts)` calculates the number of solution points that dominate
       the Nadir point, and
     * `gdPlus(pts)` calculates the RMSE where the error in each point is
       approximated by the 2-norm distance to the nearest solution point.

    """

    prob_id: int

    def __init__(self, prob_id: int, nobj: int):
        """Read the current DTLZ problem defn from environment vars."""
        super().__init__(prob_id=prob_id, nobj=nobj)

    @property
    def nadir_point(self):
        """Calculate the Nadir point for the given problem definition."""
        if self.prob_id == 1:
            return np.ones(self.nobj) * 0.5
        elif self.prob_id in [2, 3, 4, 5, 6]:
            return np.ones(self.nobj)
        elif self.prob_id == 7:
            nadir = np.ones(self.nobj)
            nadir[self.nobj - 1] = self.nobj * 2.0
            return nadir
        else:
            raise ValueError(f"DTLZ{self.prob_id} is not a valid problem")

    def num_points_dominate_nadir(self, pts):
        """Calculate the number of solutions that dominate the Nadir point.

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
        return sum([all(fi <= self.nadir_point) for fi in pareto_pts])

    def gdplus_score(self, pts):
        """Calculate the p=1 generational distance for a given solution set.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            float: The p=1 generational distance over all points in pts.

        """
        if np.any(pts < 0):
            pareto_pts = pareto_front(-pts)
        else:
            pareto_pts = pareto_front(pts)
        if self.prob_id == "1":
            dists = self._dtlz1Dist(pareto_pts)
        elif self.prob_id in ["2", "3", "4", "5", "6"]:
            dists = self._dtlz2Dist(pareto_pts)
        elif self.prob_id == "7":
            dists = self._dtlz7Dist(pareto_pts)
        else:
            raise ValueError(f"DTLZ{self.prob_id} is not a valid problem")
        return np.sum(dists) / len(dists)

    def _dtlz1Dist(self, pts):
        """Calculate the d+ from each fi to the nearest solution in DTLZ1.

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            numpy.ndarray: A 1d array of distances to the nearest solution
            point for DTLZ1.
        """
        return np.array([np.linalg.norm(np.maximum(fi - (0.5 * fi / np.sum(fi)), 0)) for fi in pts])

    def _dtlz2Dist(self, pts):
        """Calculate the d+ from each fi to the nearest point on unit sphere.

        Note: Works for DTLZ2-6

        Args:
            pts (numpy.ndarra): A 2d array of objective values.
                Each row is an objective value in the solution set.

        Returns:
            numpy.ndarray: A 1d array of distances to the surface of the
            unit sphere.

        """
        return np.array(
            [np.linalg.norm(np.maximum(fi - (fi / np.linalg.norm(fi)), 0)) for fi in pts]
        )

    def _dtlz7Dist(self, pts):
        """Calculate the d+ from each fi to the nearest soln in DTLZ7.

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
            gx = 2.0
            hx = -np.sum(
                fi[: self.nobj - 1] * (1.0 + np.sin(3.0 * np.pi * fi[: self.nobj - 1])) / gx
            ) + float(self.nobj)
            pts_proj.append(gx * hx)
        return np.array([np.abs(np.maximum(fi[-1] - fj, 0)) for fi, fj in zip(pts, pts_proj)])

    def gdplus(self, y: np.ndarray) -> np.ndarray:
        """Compute the regret of a list of given solution.

        Args:
            y (np.ndarray): An array of solutions.

        Returns:
            np.ndarray: An array of regret values.
        """
        scores = []
        for i in range(1, y.shape[0] + 1):
            scores.append(self.gdplus_score(y[:i]))
        scores = np.asarray(scores)
        return scores


class DTLZBenchmark(HPOBenchmark):
    """DTLZ benchmark.

    Args:
        nparams (int, optional): the number of parameters in the problem.
        offset (int, optional): the offset in the space of parameters.
        nslack (int, optional): the number of additional slack parameters in the problem.
    """

    def __init__(
        self,
        prob_id: int = 2,
        nparams: int = 5,
        nobj: int = 2,
        offset: Optional[int] = None,
        with_failures: bool = False,
    ) -> None:
        self.nparams = nparams
        self.nobj = nobj
        self.offset = offset
        self.with_failures = with_failures
        # Read DTLZ problem name and acquire pointer

        self.prob_id = prob_id
        self.prob_name = f"dtlz{self.prob_id}"
        self.dtlz_class = getattr(dtlz, self.prob_name)

        # Read problem dims and definition (or read from ENV)
        if self.prob_name in ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5"]:
            if self.offset is None:
                self.offset = 0.5
        elif self.prob_name in ["dtlz6", "dtlz7"]:
            if self.offset is None:
                self.offset = 0.0
        else:
            raise ValueError(f"Invalid problem {self.prob_name}")

        self.dtlz_obj = self.dtlz_class(self.nparams, self.nobj, offset=self.offset)

    @property
    def problem(self):  # noqa: D102
        problem = HpProblem()
        domain = (0.0, 1.0)
        for i in range(self.nparams):
            problem.add_hyperparameter(domain, f"x{i}")
        return problem

    @property
    def run_function(self):  # noqa: D102
        run_function_ = functools.partial(
            run_function,
            dtlz_obj=self.dtlz_obj,
            nobj=self.nobj,
            with_failures=self.with_failures,
        )
        return run_function_

    @property
    def scorer(self):  # noqa: D102
        return DTLZScorer(self.prob_id, self.nobj)
