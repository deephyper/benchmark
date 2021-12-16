import logging

from deephyper.evaluator.callback import LoggerCallback, ProfilingCallback
from deephyper.evaluator import Evaluator
from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS
import numpy as np
from deephyper_benchmark.benchmark import Benchmark
from deephyper_benchmark.run.ackley import run_ackley

logger = logging.getLogger(__name__)


class BenchmarkHPSAckley(Benchmark):
    parameters = {
        "random_state": None,
        "surrogate_model": "RF",  # RF, ET, GBRT / DUMMY
        "acq_func": "UCB",  # UCB, EI, PI, gp_hedge
        "kappa": 1.96,
        "filter_duplicated": True,
        "liar_strategy": "cl_max",  # cl_min, cl_mean, cl_max
        "n_jobs": 1,
        "evaluator_method": "ray",  # ray, process, subprocess / threadpool
        "num_workers": 1,
        "max_evals": 20,
    }

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def load_parameters(self, params):
        super().load_parameters(params)
        if self.parameters["random_state"] is None:
            self.parameters["random_state"] = np.random.randint(
                0, np.iinfo(np.int32).max
            )
        return self.parameters

    def initialize(self):
        logger.info(f"Starting initialization of *{type(self).__name__}*")

        # Creation of an hyperparameter problem
        problem = HpProblem()

        # Discrete hyperparameter (sampled with uniform prior)
        problem.add_hyperparameter((8, 128), "units")
        problem.add_hyperparameter((10, 100), "num_epochs")

        # Categorical hyperparameter (sampled with uniform prior)
        ACTIVATIONS = [
            "elu", "gelu", "hard_sigmoid", "linear", "relu", "selu",
            "sigmoid", "softplus", "softsign", "swish", "tanh",
        ]
        problem.add_hyperparameter(ACTIVATIONS, "activation")

        # Real hyperparameter (sampled with uniform prior)
        problem.add_hyperparameter((0.0, 0.6), "dropout_rate")

        # Discrete and Real hyperparameters (sampled with log-uniform)
        problem.add_hyperparameter((8, 256, "log-uniform"), "batch_size")
        problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate")

        # Add a starting point to try first
        default_config = {
            "units": 32,
            "activation": "relu",
            "dropout_rate": 0.5,
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 1e-3,
        }
        problem.add_starting_point(**default_config)

        self.profiler = ProfilingCallback()
        evaluator = Evaluator.create(
            run_ackley,
            method=self.parameters["evaluator_method"],
            method_kwargs={
                "num_workers": self.parameters["num_workers"],
                "callbacks": [LoggerCallback(), self.profiler],
            },
        )

        self.search = AMBS(
            problem,
            evaluator,
            random_state=self.parameters["random_state"],
            surrogate_model=self.parameters["surrogate_model"],
            acq_func=self.parameters["acq_func"],
            kappa=self.parameters["kappa"],
            filter_duplicated=self.parameters["filter_duplicated"],
            liar_strategy=self.parameters["liar_strategy"],
            n_jobs=self.parameters["n_jobs"],
        )
        logger.info("Finishing initialization")

    def execute(self):
        logger.info(f"Starting execution of *{type(self).__name__}*")
        self.search_result = self.search.search(max_evals=self.parameters["max_evals"])
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

        # compute best objective
        best_obj = max(search.objective)

        # return results
        self.results["search"] = search.to_dict(orient="list")
        self.results["profile"] = profile.to_dict(orient="list")
        self.results["best_obj"] = best_obj
        self.results["perc_util"] = perc_util

        return self.results
