import logging

import numpy as np
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback, ProfilingCallback
from deephyper.search.hps import AMBS
from deephyper.sklearn.classifier import problem_autosklearn1
from deephyper_benchmark.benchmark import Benchmark
from scripts.local.run_functions.run_functions import run_sleep

logger = logging.getLogger(__name__)


class BenchmarkHPSAMBSOnCustomSleep(Benchmark):
    parameters = {
        "run_duration": 1,
        "num_workers": 4,
        "max_evals": 8,
        "random_duration": False
    }

    def __init__(self, verbose=0) -> None:
        super().__init__(verbose=verbose)
        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def load_parameters(self, params) -> dict:
        super().load_parameters(params)
        return self.parameters

    def initialize(self) -> None:
        logger.info(f"Starting initialization of *{type(self).__name__}*")

        self.problem = problem_autosklearn1
        self.problem.add_hyperparameter((self.parameters["run_duration"], self.parameters["run_duration"]+0.1), "run_duration", self.parameters["run_duration"])
        self.problem.add_hyperparameter([self.parameters["random_duration"]], "random_duration", self.parameters["random_duration"])

        self.profiler = ProfilingCallback()
        self.evaluator = Evaluator.create(
            run_sleep,
            method="ray",
            method_kwargs={
                "num_workers": self.parameters["num_workers"],
                "callbacks": [LoggerCallback(), self.profiler],
            },
        )

        self.search = AMBS(
            self.problem,
            self.evaluator,
            random_state=42,
            surrogate_model="RF",
            acq_func="UCB",
            kappa=1.96,
            filter_duplicated=True,
            liar_strategy="cl_max",
            n_jobs=1,
        )
        logger.info("Finishing initialization")

    def execute(self) -> None:
        logger.info(f"Starting execution of *{type(self).__name__}*")
        self.search.search(max_evals=self.parameters["max_evals"])
        self.profile_result = self.profiler.profile

    def report(self) -> dict:
        logger.info(f"Starting the report of *{type(self).__name__}*")

        num_workers = self.parameters["num_workers"]
        profile = self.profile_result

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
        self.results["profile"] = profile.to_dict(orient="list")
        self.results["perc_util"] = perc_util

        return self.results
