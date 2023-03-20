# Setup info-level logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - " + \
           "%(message)s",
    force=True,
)

# Set DTLZ problem environment variables
import os
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = "5" # 5 vars
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = "3" # 2 objs
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "2" # DTLZ2 problem
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.6" # [x_o, .., x_d]*=0.6

# Load DTLZ benchmark suite
import deephyper_benchmark as dhb
# dhb.install("DTLZ")
dhb.load("DTLZ")


# Necessary IF statement otherwise it will enter in a infinite loop
# when loading the 'run' function from a subprocess
if __name__ == "__main__":
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    # Run HPO-pipeline with default configuration of hyperparameters
    from deephyper_benchmark.lib.dtlz import hpo
    from deephyper.evaluator import RunningJob, Evaluator
    config = hpo.problem.default_configuration
    print(config)
    res = hpo.run(RunningJob(parameters=config))
    print(f"{res=}")

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        hpo.run,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define your search and execute it
    search = CBO(hpo.problem, evaluator)

    # solve with 100 evals
    results = search.search(max_evals=100)
    print(results)
