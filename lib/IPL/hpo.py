import os

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import profile, RunningJob
from deephyper.problem import HpProblem


problem = HpProblem()
problem.add_hyperparameter((1e-6, 10, "log-uniform"), "alpha")
problem.add_hyperparameter((1.0, 10.0), "beta")
problem.add_hyperparameter((0.1, 0.5), "gamma")

# optimum is at
# C(s=100, alpha=1e-6, beta=1.0, gamma=0.5) -> 0.100001

def ipl_curve(s, alpha, beta, gamma):
    """Inverse Power-Law Model"""
    return alpha + beta * s**-gamma

@profile
def run(job: RunningJob, optuna_trial=None) -> dict:

    # otherwise failure
    config = job.parameters
    min_b, max_b = 1, 100

    alpha = job.parameters["alpha"]
    beta = job.parameters["beta"]
    gamma = job.parameters["gamma"]

    if optuna_trial:

        for budget_i in range(min_b, max_b + 1):
            objective_i = -ipl_curve(budget_i, alpha, beta, gamma) # maximisation in deephyper
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        return {
            "objective": objective_i,
            "metadata": {"budget": budget_i, "stopped": budget_i < max_b},
        }

    else:

        for budget_i in range(min_b, max_b + 1):
            objective_i = -ipl_curve(budget_i, alpha, beta, gamma) # maximisation in deephyper
            job.record(budget_i, objective_i)
            if job.stopped():
                break

        return {
            "objective": job.observations,
            "metadata": {"budget": budget_i, "stopped": budget_i < max_b},
        }


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
