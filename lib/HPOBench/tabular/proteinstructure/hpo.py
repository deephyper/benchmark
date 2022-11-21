import os

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import profile, RunningJob
from deephyper.problem import HpProblem

# hpobench
from hpobench.benchmarks.nas.tabular_benchmarks import ProteinStructureBenchmark

data_path = os.path.join(DIR, "../build/HPOBench/data/fcnet_tabular_benchmarks/")
b = ProteinStructureBenchmark(data_path=data_path)
config_space = b.get_configuration_space()

problem = HpProblem(config_space=config_space)


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:

    # otherwise failure
    config = job.parameters

    from hpobench.benchmarks.nas.tabular_benchmarks import ProteinStructureBenchmark

    b = ProteinStructureBenchmark(data_path=data_path)
    budget_space = b.get_fidelity_space().get("budget")
    min_b, max_b = budget_space.lower, budget_space.upper

    if optuna_trial:

        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(config, fidelity={"budget": budget_i})
            objective_i = -eval["function_value"]  # maximizing in deephyper
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        return {
            "objective": objective_i,
            "metadata": {"budget": budget_i, "stopped": budget_i < max_b},
        }

    else:

        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(config, fidelity={"budget": budget_i})
            objective_i = -eval["function_value"]  # maximizing in deephyper
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
