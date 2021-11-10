import logging

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback, ProfilingCallback
from deephyper.search.hps import AMBS
from deephyper.sklearn.classifier import problem_autosklearn1
from deephyper_benchmark.benchmark import Benchmark
from scripts.local.run_functions import run_diabetes

logger = logging.getLogger(__name__)


class BenchmarkHPSAMBSOnAutoSKLearn(Benchmark):
    parameters = {
        "random_state": 42,
        "surrogate_model": "RF",  # RF, ET, GBRT / DUMMY
        "acq_func": "UCB",  # UCB, EI, PI, gp_hedge
        "kappa": 1.96,
        "filter_duplicated": True,
        "liar_strategy": "cl_max",  # cl_min, cl_mean, cl_max
        "n_jobs": 1,
        "evaluator_method": "ray",  # ray, process, subprocess / threadpool
        "num_workers": 1,
        "max_evals": 100
    }

    def __init__(self, verbose=0) -> None:
        super().__init__(verbose=verbose)
        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def initialize(self) -> None:
        logger.info(f"Starting initialization of *{type(self).__name__}*")

        logger.info("Creating the problem...")
        self.problem = problem_autosklearn1

        logger.info("Creating the evaluator...")
        self.profiler = ProfilingCallback()
        self.evaluator = Evaluator.create(
            run_diabetes,
            method=self.parameters["evaluator_method"],
            method_kwargs={
                "num_workers": self.parameters["num_workers"],
                "callbacks": [LoggerCallback(), self.profiler]
            }
        )
        logger.info(
            f"Evaluator created with {self.evaluator.num_workers} worker{'s' if self.evaluator.num_workers > 1 else ''}")

        logger.info("Creating the search...")
        self.search = AMBS(
            self.problem,
            self.evaluator,
            random_state=self.parameters["random_state"],
            surrogate_model=self.parameters["surrogate_model"],
            acq_func=self.parameters["acq_func"],
            kappa=self.parameters["kappa"],
            filter_duplicated=self.parameters["filter_duplicated"],
            liar_strategy=self.parameters["liar_strategy"],
            n_jobs=self.parameters["n_jobs"]
        )
        logger.info("Finishing initialization")

    def execute(self) -> None:
        logger.info(f"Starting execution of *{type(self).__name__}*")

        self.search_result = self.search.search(
            max_evals=self.parameters["max_evals"])
        self.profile_result = self.profiler.profile

    def report(self) -> dict:
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

        # compute best objective
        best_obj = max(search.objective)

        # return results
        self.results["search"] = search.to_dict(orient='list')
        self.results["profile"] = profile.to_dict(orient='list')
        self.results["best_obj"] = best_obj
        self.results["perc_util"] = perc_util

        return self.results
