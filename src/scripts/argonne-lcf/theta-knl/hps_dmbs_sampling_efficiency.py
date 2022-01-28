import logging
import time
import pandas as pd

import ray

from deephyper.evaluator.callback import ProfilingCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import DMBS
from deephyper_benchmark.benchmark import Benchmark

import deephyper_benchmark.run.run_ackley as run_ackley

logger = logging.getLogger(__name__)


class BenchmarkHPSDMBSamplingEfficiency(Benchmark):
    parameters = {
        "num_dimensions": 1,
        "num_workers": 1,
        "timeout": 5,
        "sleep_duration": 0,
        "sleep_duration_noise": 0,
        "n_points": 10000,
        "n_jobs": 1,
    }

    def __init__(self, verbose=0):
        super().__init__(verbose)
        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def load_parameters(self, params):
        super().load_parameters(params)

        err_msg = "{}: should be {}, but found '{}'"

        assert (
            self.parameters["num_dimensions"] >= 1
            and type(self.parameters["num_dimensions"]) is int
        )
        assert self.parameters["num_workers"] > 0, err_msg.format(
            "num_workers", "positive", self.parameters["num_workers"]
        )
        assert self.parameters["timeout"] >= 0, err_msg.format(
            "timeout", "positive", self.parameters["timeout"]
        )
        assert self.parameters["sleep_duration"] >= 0, err_msg.format(
            "sleep_duration", "positive", self.parameters["sleep_duration"]
        )
        assert self.parameters["sleep_duration_noise"] >= 0, err_msg.format(
            "sleep_duration_noise", "positive", self.parameters["sleep_duration_noise"]
        )
        assert self.parameters["n_points"] >= 0, err_msg.format(
            "n_points", "n_points", self.parameters["n_points"]
        )

        return self.parameters

    def initialize(self):
        logger.info(f"Starting initialization of *{type(self).__name__}*")

        logger.info("Creating the problem...")
        self.problem = HpProblem()
        for i in range(self.parameters["num_dimensions"]):
            self.problem.add_hyperparameter((-32.768, 32.768), f"x{i}")

        # initialize ray
        ray.init(address="auto")
        self.parameters["num_workers"] = int(
            sum([node["Resources"].get("CPU", 0) for node in ray.nodes()])
        ) - 1

        logger.info("Creating the search...")
        self.search = DMBS(
            self.problem,
            run_ackley.run,
            run_function_kwargs={
                "sleep_duration": self.parameters["sleep_duration"],
                "sleep_duration_noise": self.parameters["sleep_duration_noise"],
            },
            num_workers=self.parameters["num_workers"],
        )

    def execute(self):
        logger.info(f"Starting execution of *{type(self).__name__}*")

        self.search_result = self.search.search(
            max_evals=-1, timeout=self.parameters["timeout"]
        )

    def report(self):
        logger.info(f"Starting the report of *{type(self).__name__}*")

        num_workers = self.parameters["num_workers"]
        search = self.search_result

        # generating the profile
        jobs_start = [(t, 1) for t in search['timestamp_start']]
        jobs_end = [(t, -1) for t in search['timestamp_end']]
        history = sorted(jobs_start + jobs_end)
        n_jobs = 0
        profile = []
        for t, incr in sorted(history):
            n_jobs += incr
            profile.append([t, n_jobs])
        cols = ["timestamp", "n_jobs_running"]
        profile = pd.DataFrame(profile, columns=cols)

        # compute worker utilization
        t0 = 0
        t_max = profile.iloc[-1].timestamp
        T_max = (t_max - t0) * num_workers

        cum = 0
        for i in range(len(profile.timestamp) - 1):
            cum += (
                profile.timestamp.iloc[i + 1] - profile.timestamp.iloc[i]
            ) * profile.n_jobs_running.iloc[i]
        perc_util = cum / T_max

        best_obj = max(search.objective)

        # return results
        self.results["perc_util"] = perc_util
        self.results["profile"] = {"data": profile.to_dict(orient="list"), "num_workers": self.parameters["num_workers"]}
        self.results["search"] = search.to_dict(orient="list")
        self.results["best_obj"] = best_obj
        self.results["nb_iter"] = len(search.index)

        return self.results
