import os

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import profile
from deephyper.problem import HpProblem

# hpobench
from hpobench.benchmarks.nas.tabular_benchmarks import ParkinsonsTelemonitoringBenchmark

data_path = os.path.join(DIR, "../build/HPOBench/data/fcnet_tabular_benchmarks/")
b = ParkinsonsTelemonitoringBenchmark(data_path=data_path)
config_space = b.get_configuration_space()

problem = HpProblem(config_space=config_space)


@profile
def run(config, optuna_trial=None):
    # otherwise failure
    config.pop("job_id", None)
    
    from hpobench.benchmarks.nas.tabular_benchmarks import ParkinsonsTelemonitoringBenchmark

    b = ParkinsonsTelemonitoringBenchmark(data_path=data_path)

    if optuna_trial:
        budget_space = b.get_fidelity_space().get("budget")
        min_b, max_b = budget_space.lower, budget_space.upper
        pruned = False
        for budget_i in range(min_b, max_b):
            eval = b.objective_function(config, fidelity=budget_i)
            objective_i = -eval["function_value"] # maximizing in deephyper
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                pruned = True
                break
        return {"objective": objective_i, "budget": budget_i, "pruned": pruned}
    else:
        eval = b.objective_function(config)
        objective = -eval["function_value"] # maximizing in deephyper
        return objective


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(default_config)
    print(f"{result=}")
