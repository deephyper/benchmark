import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob
from . import model


# Create problem
problem = HpProblem()
jahs_obj = model.jahs_bench()


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
