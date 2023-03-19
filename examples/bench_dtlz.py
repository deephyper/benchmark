# Setup info-level logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

# Set DTLZ problem environment variables
import os
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = "5"
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = "3"
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "2"
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.6"

# Load DTLZ benchmark suite
import deephyper_benchmark as dhb
# dhb.install("DTLZ")
dhb.load("DTLZ")

# Run HPO-pipeline with default configuration of hyperparameters
from deephyper_benchmark.lib.dtlz import hpo
from deephyper.evaluator import RunningJob
config = hpo.problem.default_configuration
print(config)
res = hpo.run(RunningJob(parameters=config))
print(f"{res=}")


