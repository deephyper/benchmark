import logging

from deephyper.evaluator.callback import ProfilingCallback
from deephyper.evaluator import Evaluator
from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS
from deephyper_benchmark.benchmark import Benchmark
from deephyper_benchmark.run.run_ackley import run

logger = logging.getLogger(__name__)


class BenchmarkTest1(Benchmark):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def load_parameters(self, **kwargs):
        err_msg = "'{}' is not found in the {}"
        assert "num_workers" in kwargs.keys(), err_msg.format("num_workers", "parameters")
        assert "evaluator_method" in kwargs, err_msg.format("evaluator_method", "parameters")
        assert "max_evals" in kwargs, err_msg.format("max_evals", "parameters")
        assert "starting_point" in kwargs, err_msg.format("starting_point", "parameters")

        self.parameters = {
            "num_workers": kwargs["num_workers"],
            "evaluator_method": kwargs["evaluator_method"],
            "max_evals": kwargs["max_evals"],
            "starting_point": tuple(kwargs["starting_point"])
        }

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
            method_kwargs={"num_workers": self.parameters["num_workers"], "callbacks": [self.profiler]},
        )
        logger.info(
            f"Evaluator created with {self.evaluator.num_workers} worker{'s' if self.evaluator.num_workers > 1 else ''}"
        )

        logger.info("Creating the search...")
        self.search = AMBS(self.problem, self.evaluator)

        logger.info("Finishing initialization")

    def execute(self):
        logger.info(f"Starting execution of *{type(self).__name__}*")

        self.results["search"] = self.search.search(max_evals=self.parameters["max_evals"])
        self.results["profile"] = self.profiler.profile

    def report(self):
        logger.info(f"Starting the report of *{type(self).__name__}*")
        
        err_msg = "'{}' is not found in the {}"
        assert "profile" in self.results, err_msg.format("profile", "report")
        assert "search" in self.results, err_msg.format("search", "report")
        assert "init_time" in self.results, err_msg.format("init_time", "report")
        assert "exec_time" in self.results, err_msg.format("exec_time", "report")

        num_workers = self.parameters["num_workers"]
        profile = self.results["profile"]
        search = self.results["search"]

        # keys of profile: timestamp n_jobs_running
        assert "timestamp" in profile.columns, err_msg.format("timestamp", "profile")
        assert "n_jobs_running" in profile.columns, err_msg.format("n_jobs_running", "profile")

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
        self.results["perc_util"] = perc_util
        
        t0 = profile.iloc[0].timestamp
        profile.timestamp -= t0
        
        # saving report
        self.results["profile"] = profile.to_dict(orient='list')
        self.results["search"] = search.to_dict(orient='list')

        report = {
            "parameters": self.parameters,
            "results": self.results
            }

        return report
