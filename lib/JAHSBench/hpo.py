import os
import numpy as np
import time

from deephyper.evaluator import profile, RunningJob
from deephyper.problem import HpProblem
from . import model


# Read in whether to do single- or multi-objectives
multiobj = int(os.environ.get("DEEPHYPER_BENCHMARK_MOO", 1))
prob_name = os.environ.get("DEEPHYPER_BENCHMARK_JAHS_PROB", "fashion_mnist")

# Create problem
problem = HpProblem()
jahs_obj = model.jahs_bench(dataset=prob_name)
# 2 continuous hyperparameters
problem.add_hyperparameter((1.0e-3, 1.0), "LearningRate")
problem.add_hyperparameter((1.0e-5, 1.0e-3), "WeightDecay")
# 2 categorical hyperparameters
problem.add_hyperparameter(["ReLU", "Hardswish", "Mish"], "Activation")
problem.add_hyperparameter(["on", "off"], "TrivialAugment")
# 6 categorical architecture design variables
for i in range(1, 7):
    problem.add_hyperparameter([0, 1, 2, 3, 4], f"Op{i}")
# 1 integer hyperparameter number of training epochs (1 to 200)
problem.add_hyperparameter((1, 200), "nepochs")

@profile
def run(job: RunningJob, sleep=False, sleep_scale=0.01) -> dict:

    config = job.parameters
    result = jahs_obj(config)

    if sleep:
        t_sleep = result["runtime"] * sleep_scale
        time.sleep(t_sleep)

    dh_data = {}
    dh_data["metadata"] = result
    if multiobj:
        dh_data["objective"] = [
                                result["valid-acc"],
                                -result["latency"],
                                -result['size_MB']
                               ]
    else:
        dh_data["objective"] = result["valid-acc"]
    return dh_data


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
