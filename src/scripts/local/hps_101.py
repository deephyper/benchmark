import logging

from deephyper.evaluator.callback import ProfilingCallback
from deephyper.evaluator import Evaluator
from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS
from deephyper_benchmark.benchmark import Benchmark
from deephyper_benchmark.run.run_ackley import run

logger = logging.getLogger(__name__)


class BenchmarkHPS101(Benchmark):
    parameters = {
        "num_workers": 6,
        "evaluator_method": "ray",
        "max_evals": 10,
        "starting_point": [-32.768, 32.768]
    }

    def __init__(self, verbose=0):
        super().__init__(verbose)

        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def load_parameters(self, params):
        super().load_parameters(params)

        err_msg = "{}: should be {}, but found '{}'"
        assert self.parameters["evaluator_method"] in ["process", "subprocess", "ray"], err_msg.format(
            "evaluator_method", "either process, subprocess, or ray", self.parameters["evaluator_method"])
        assert self.parameters["num_workers"] > 0, err_msg.format(
            "num_workers", "positive", self.parameters["num_workers"])
        assert self.parameters["max_evals"] > 0, err_msg.format(
            "max_evals", "positive", self.parameters["max_evals"])

        self.parameters["starting_point"] = tuple(
            self.parameters["starting_point"])

        return self.parameters

    def initialize(self):
        logger.info(f"Starting initialization of *{type(self).__name__}*")

        logger.info("Creating the problem...")
        self.problem = HpProblem()
        self.problem.add_hyperparameter(self.parameters["starting_point"], "x")

        logger.info("Creating the evaluator...")
        self.profiler = ProfilingCallback()
        self.evaluator = Evaluator.create(
            run,
            method=self.parameters["evaluator_method"],
            method_kwargs={
                "num_workers": self.parameters["num_workers"], "callbacks": [self.profiler]},
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
            max_evals=self.parameters["max_evals"])
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
        for i in range(len(profile.timestamp)-1):
            cum += (
                profile.timestamp.iloc[i + 1] - profile.timestamp.iloc[i]
            ) * profile.n_jobs_running.iloc[i]
        perc_util = cum / T_max

        t0 = profile.iloc[0].timestamp
        profile.timestamp -= t0

        # return results
        self.results["perc_util"] = perc_util
        self.results["profile"] = profile.to_dict(orient='list')
        self.results["search"] = search.to_dict(orient='list')

        return self.results
