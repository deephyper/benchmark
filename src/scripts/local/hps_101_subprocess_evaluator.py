import logging

from deephyper.evaluator.callback import ProfilingCallback
from deephyper.evaluator import Evaluator
from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS
from deephyper_benchmark.benchmark import Benchmark
from deephyper_benchmark.run.run_ackley import run

logger = logging.getLogger(__name__)


class HPS101SubprocessEvaluator(Benchmark):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def initialize(self):
        logger.info(f"Starting initialization of *{type(self).__name__}*")

        logger.info("Creating the problem...")
        self.problem = HpProblem()
        self.problem.add_hyperparameter((-32.768, 32.768), "x")

        logger.info("Creating the evaluator...")
        self.profiler = ProfilingCallback()
        self.evaluator = Evaluator.create(
            run,
            method="subprocess",
            method_kwargs={"num_workers": 6, "callbacks": [self.profiler]},
        )
        logger.info(
            f"Evaluator created with {self.evaluator.num_workers} worker{'s' if self.evaluator.num_workers > 1 else ''}"
        )

        logger.info("Creating the search...")
        self.search = AMBS(self.problem, self.evaluator)

        self.results = {
            "num_workers": self.evaluator.num_workers
        }

        logger.info("Finishing initialization")

    def execute(self):
        logger.info(f"Starting execution of *{type(self).__name__}*")

        self.results["search"] = self.search.search(max_evals=1000)
        self.results["profile"] = self.profiler.profile

    def report(self):
        logger.info(f"Starting the report of *{type(self).__name__}*")

        return self.results
