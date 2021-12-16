import logging
import time

from deephyper.evaluator.callback import ProfilingCallback
from deephyper.evaluator import Evaluator
from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS
from deephyper_benchmark.benchmark import Benchmark

import deephyper_benchmark.run.run_ackley as run_ackley

logger = logging.getLogger(__name__)


class BenchmarkHPSAMBSSamplingEfficiency(Benchmark):
    parameters = {
        "num_dimensions": 1,
        "num_workers": 1,
        "timeout": 5,
        "sleep_duration": 0,
        "sleep_duration_noise": 0,
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

        return self.parameters

    def initialize(self):
        logger.info(f"Starting initialization of *{type(self).__name__}*")

        logger.info("Creating the problem...")
        self.problem = HpProblem()
        for i in range(self.parameters["num_dimensions"]):
            self.problem.add_hyperparameter((-32.768, 32.768), f"x{i}")

        logger.info("Creating the evaluator...")
        self.profiler = ProfilingCallback()
        self.evaluator = Evaluator.create(
            run_ackley.run,
            method="ray",
            method_kwargs={
                "num_workers": self.parameters["num_workers"],
                "callbacks": [self.profiler],
                "run_function_kwargs": {
                    "sleep_duration": self.parameters["sleep_duration"],
                    "sleep_duration_noise": self.parameters["sleep_duration_noise"],
                },
            },
        )
        logger.info(
            f"Evaluator created with {self.evaluator.num_workers} worker{'s' if self.evaluator.num_workers > 1 else ''}"
        )

        logger.info("Creating the search...")
        self.search = AMBS(self.problem, self.evaluator)

        logger.info("Finishing initialization")

    def execute(self):
        logger.info(f"Starting execution of *{type(self).__name__}*")

        self.search_result = self.search.search(
            max_evals=-1, timeout=self.parameters["timeout"]
        )
        self.profile_result = self.profiler.profile

    def report(self):
        logger.info(f"Starting the report of *{type(self).__name__}*")

        num_workers = self.parameters["num_workers"]
        profile = self.profile_result
        search = self.search_result

        # compute worker utilization
        t0 = profile.iloc[0].timestamp
        t_max = profile.iloc[-1].timestamp
        T_max = (t_max - t0) * num_workers

        cum = 0
        for i in range(len(profile.timestamp) - 1):
            cum += (
                profile.timestamp.iloc[i + 1] - profile.timestamp.iloc[i]
            ) * profile.n_jobs_running.iloc[i]
        perc_util = cum / T_max

        t0 = profile.iloc[0].timestamp
        profile.timestamp -= t0

        # return results
        self.results["perc_util"] = perc_util
        self.results["profile"] = profile.to_dict(orient="list")
        self.results["search"] = search.to_dict(orient="list")

        return self.results