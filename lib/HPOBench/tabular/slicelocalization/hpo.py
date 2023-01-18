import os

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import profile, RunningJob
from deephyper.problem import HpProblem

# hpobench
from hpobench.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark

data_path = os.path.join(DIR, "../build/HPOBench/data/fcnet_tabular_benchmarks/")
b = SliceLocalizationBenchmark(data_path=data_path)
config_space = b.get_configuration_space()

problem = HpProblem(config_space=config_space)


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:

    # otherwise failure
    config = job.parameters

    from hpobench.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark

    b = SliceLocalizationBenchmark(data_path=data_path)
    budget_space = b.get_fidelity_space().get("budget")
    min_b, max_b = budget_space.lower, budget_space.upper

    eval_test = b.objective_function_test(config, fidelity={"budget": max_b})
    objective_test = -eval_test["function_value"]

    other_metadata = {}

    if optuna_trial:

        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(config, fidelity={"budget": budget_i})
            objective_i = -eval["function_value"]  # maximizing in deephyper
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        objective = objective_i

    else:

        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(config, fidelity={"budget": budget_i})
            objective_i = -eval["function_value"]  # maximizing in deephyper
            job.record(budget_i, objective_i)
            if job.stopped():
                break

        objective = job.objective

        if hasattr(job, "stopper") and hasattr(job.stopper, "infos_stopped"):
            other_metadata["infos_stopped"] = job.stopper.infos_stopped

    metadata = {
        "budget": budget_i,
        "stopped": budget_i < max_b,
        "objective_test": objective_test,
    }
    metadata.update(other_metadata)
    return {
        "objective": objective,
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
