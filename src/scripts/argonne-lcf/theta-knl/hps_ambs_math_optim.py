import logging
import pathlib
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deephyper.evaluator.callback import ProfilingCallback
from deephyper.evaluator import Evaluator
from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS
from deephyper_benchmark.benchmark import Benchmark
from deephyper_benchmark.run.math import ackley, bohachevsky, branin, eggholder, hartmann6D, rosenbrock, schubert

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI
if not MPI.Is_initialized():
    MPI.Init_thread()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger(__name__)


class BenchmarkHPSAMBSMathOptim(Benchmark):
    parameters = {
        "math_function": "ackley",
        "liar_strategy": "boltzmann",
        "timeout": 5,
    }

    math_functions = {
        "ackley": ackley,
        "bohachevsky": bohachevsky,
        "branin": branin,
        "eggholder": eggholder,
        "hartmann6D": hartmann6D,
        "rosenbrock": rosenbrock,
        "schubert": schubert,
    }
    input_space_dims = {
        "ackley": 15,
        "bohachevsky": 2,
        "branin": 2,
        "eggholder": 2,
        "hartmann6D": 6,
        "rosenbrock": 4,
        "schubert": 2,
    }
    input_space_restrict = {
        "ackley": (-32.768, 32.768)
        "bohachevsky": (-100.0, 100.0)
        "branin": (0.0, 1.0)
        "eggholder": (-512.0, 512.0)
        "hartmann6D": (0.0, 1.0)
        "rosenbrock": (0.0, 1.0)
        "schubert": (-10.0, 10.0)
    }

    def __init__(self, verbose=0):
        super().__init__(verbose)
        if self.verbose:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    def load_parameters(self, params):
        super().load_parameters(params)

        err_msg = "{}: should be {}, but found '{}'"

        assert self.parameters["math_function"] >= 0, err_msg.format(
            "math_function", f"in {math_functions.keys()}", self.parameters["math_function"]
        )
        assert self.parameters["liar_strategy"] >= 0, err_msg.format(
            "liar_strategy", "in boltzmann, cl_max, cl_min, cl_mean", self.parameters["liar_strategy"]
        )
        assert self.parameters["timeout"] >= 0, err_msg.format(
            "timeout", "positive", self.parameters["timeout"]
        )

        return self.parameters

    def initialize(self):
        logger.info(f"Starting initialization of *{type(self).__name__}*")
        if rank == 0:
            log_dir = f"/dev/shm/logs"
            pathlib.Path(log_dir).mkdir(parents=False, exist_ok=False)

            log_file = f"{log_dir}/deephyper.log"
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            )

            math_function = math_functions.get(self.parameters["math_function"])
            nb_dim = input_space_dims.get(self.parameters["math_function"])
            domain = input_space_restrict.get(self.parameters["math_function"])

            logger.info("Creating the problem...")
            self.problem = HpProblem()
            for i in range(nb_dim):
                self.problem.add_hyperparameter(domain, f"x{i}")
            

        logger.info("Creating the evaluator...")
        self.profiler = ProfilingCallback()
        self.evaluator = Evaluator.create(
            math_function.run,
            method="mpicomm",
            method_kwargs={
                "callbacks": [self.profiler],
            },
        )
        if self.evaluator is not None:
            logger.info(
                f"Evaluator created with {self.evaluator.num_workers} worker{'s' if self.evaluator.num_workers > 1 else ''}"
            )
            self.parameters["num_workers"] = self.evaluator.num_workers

            logger.info("Creating the search...")
            self.search = AMBS(
                self.problem,
                self.evaluator,
                log_dir=log_dir,
                liar_strategy=self.parameters["liar_strategy"],
            )

    def execute(self):
        logger.info(f"Starting execution of *{type(self).__name__}*")

        if self.evaluator is not None:
            self.search_result = self.search.search(
                max_evals=-1, timeout=60 * self.parameters["timeout"]
            )

    def report(self):
        logger.info(f"Starting the report of *{type(self).__name__}*")

        num_workers = self.parameters["num_workers"]
        profile = self.profiler.profile
        search = self.search_result

        # compute worker utilization
        t_max = profile.iloc[-1].timestamp
        T_max = t_max * num_workers

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

        if rank != 0:
            self.results = None
        return self.results
