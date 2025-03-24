"""here."""

import numpy as np
import time

from deephyper.evaluator import profile
from deephyper.evaluator import RunningJob


@profile
def run_function(job: RunningJob, bb_func, sleep=False, sleep_mean=60, sleep_noise=20) -> dict:  # noqa: D103
    config = job.parameters

    if sleep:
        t_sleep = np.random.normal(loc=sleep_mean, scale=sleep_noise)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

    return bb_func(x)
