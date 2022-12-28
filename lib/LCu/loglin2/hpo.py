import os

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import profile, RunningJob
from deephyper.problem import HpProblem


problem = HpProblem()
problem.add_hyperparameter((1e-6, 10), "rho_0")
problem.add_hyperparameter((1.0, 10.0), "rho_1")


def f_loglin2(b, rho):
    return np.power(b, rho[1]) * np.exp(rho[0])


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:

    print(f"{job.id=}")

    # otherwise failure
    min_b, max_b = 1, 100

    rho = [job.parameters["rho_0"], job.parameters["rho_1"]]
    f = lambda b: f_loglin2(b, rho)

    objective_test = f(max_b)

    if optuna_trial:

        for budget_i in range(min_b, max_b + 1):
            objective_i = f(budget_i)
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        return {
            "objective": objective_i,
            "metadata": {
                "budget": budget_i,
                "stopped": budget_i < max_b,
                "objective_test": objective_test,
            },
        }

    else:

        for budget_i in range(min_b, max_b + 1):
            objective_i = f(budget_i)
            job.record(budget_i, objective_i)
            if job.stopped():
                break

        return {
            "objective": job.observations,
            "metadata": {
                "budget": budget_i,
                "stopped": budget_i < max_b,
                "objective_test": objective_test,
            },
        }


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
