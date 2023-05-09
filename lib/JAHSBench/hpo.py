import time
import numpy as np
from deephyper.problem import NaProblem
from deephyper.evaluator import profile, RunningJob
from . import model


# Create problem
problem = NaProblem()
jahs_obj = model.jahs_bench()

problem.hyperparameter(
        LearningRate=(1.0e-3, 1.0), "LearningRate")
problem.add_hyperparameter((1.0e-5, 1.0e-3), "WeightDecay")
# 2 categorical variables
moop_rbf.addDesign({'name': "Activation",
                    'des_type': "categorical",
                    'levels': ["ReLU", "Hardswish", "Mish"]})
moop_rbf.addDesign({'name': "TrivialAugment",
                    'des_type': "categorical",
                    'levels': ["on", "off"]})
# 6 integer variables
for i in range(1, 7):
    moop_rbf.addDesign({'name': f"Op{i}",
                        'des_type': "integer",
                        'lb': 0, 'ub': 4})

@profile
def run(job: RunningJob, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:

    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    f1, f2 = jahs_obj(x)

    return f1, -f2


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
