import os
import time

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import profile, RunningJob
from deephyper.problem import HpProblem

simulate_run_time = bool(int(os.environ.get("DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME", 0)))
prop_real_run_time = float(os.environ.get("DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME", 1.0))

# hpobench
from hpobench.benchmarks.nas.tabular_benchmarks import NavalPropulsionBenchmark

data_path = os.path.join(DIR, "../build/HPOBench/data/fcnet_tabular_benchmarks/")
b = NavalPropulsionBenchmark(data_path=data_path)
config_space = b.get_configuration_space()

problem = HpProblem(config_space=config_space)


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:

    # otherwise failure
    config = job.parameters

    from hpobench.benchmarks.nas.tabular_benchmarks import NavalPropulsionBenchmark

    b = NavalPropulsionBenchmark(data_path=data_path)
    budget_space = b.get_fidelity_space().get("budget")
    min_b, max_b = budget_space.lower, budget_space.upper

    eval_test = b.objective_function_test(config, fidelity={"budget": max_b})
    objective_test = -eval_test["function_value"]

    eval_val = b.objective_function(config, fidelity={"budget": max_b})
    objective_val = -eval_val["function_value"]

    other_metadata = {}

    if optuna_trial:

        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(config, fidelity={"budget": budget_i})

            cost_step = eval["cost"] -  consumed_time
            consumed_time = eval["cost"]
            if simulate_run_time:
                time.sleep(cost_step * prop_real_run_time)

            objective_i = -eval["function_value"]  # maximizing in deephyper
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        objective = objective_i

    else:

        consumed_time = 0 # the cost here corresponds to time
        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(config, fidelity={"budget": budget_i})

            cost_step = eval["cost"] -  consumed_time
            consumed_time = eval["cost"]
            if simulate_run_time:
                time.sleep(cost_step * prop_real_run_time)

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
        "objective_val": objective_val,
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
