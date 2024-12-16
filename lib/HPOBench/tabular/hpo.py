import os
import time

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

import hpobench.benchmarks.nas.tabular_benchmarks

from deephyper.evaluator import profile, RunningJob
from deephyper.hpo import HpProblem

DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME = bool(
    int(os.environ.get("DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME", 0))
)
DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME = float(
    os.environ.get("DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME", 1.0)
)
DEEPHYPER_BENCHMARK_TASK = os.environ.get("DEEPHYPER_BENCHMARK_TASK", "navalpropulsion")
DEEPHYPER_BENCHMARK_MOO = bool(int(os.environ.get("DEEPHYPER_BENCHMARK_MOO", 0)))


map_task_to_benchmark = {
    "navalpropulsion": "NavalPropulsionBenchmark",
    "parkinsonstelemonitoring": "ParkinsonsTelemonitoringBenchmark",
    "proteinstructure": "ProteinStructureBenchmark",
    "slicelocalization": "SliceLocalizationBenchmark",
}
data_path = os.path.join(DIR, "build/HPOBench/data/fcnet_tabular_benchmarks/")


benchmark_class = getattr(
    hpobench.benchmarks.nas.tabular_benchmarks,
    map_task_to_benchmark[DEEPHYPER_BENCHMARK_TASK],
)
b = benchmark_class(data_path=data_path)
config_space = b.get_configuration_space()

problem = HpProblem(config_space=config_space)


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:
    # otherwise failure
    config = job.parameters

    seed = int(job.id.split(".")[-1])
    rng = np.random.RandomState(seed)
    run_index = int(rng.choice((0, 1, 2, 3)))

    benchmark_class = getattr(
        hpobench.benchmarks.nas.tabular_benchmarks,
        map_task_to_benchmark[DEEPHYPER_BENCHMARK_TASK],
    )
    b = benchmark_class(data_path=data_path)
    budget_space = b.get_fidelity_space().get("budget")
    min_b, max_b = budget_space.lower, budget_space.upper

    # The objective is the coefficient of determination (R^2)
    # As the target were standardized, the R^2 = 1 - MSE / MSE_baseline with MSE_baseline = 1.0
    # The objective is maximized in deephyper
    eval_test = b.objective_function_test(config, fidelity={"budget": max_b})
    objective_test = 1 - eval_test["function_value"]
    cost_eval = eval_test["cost"]
    cost_step = cost_eval / (max_b - min_b + 1)

    eval_val = b.objective_function(
        config, fidelity={"budget": max_b}, run_index=run_index
    )
    objective_val = 1 - eval_val["function_value"]

    other_metadata = {}

    consumed_time = 0  # the cost here corresponds to time

    if optuna_trial:
        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(
                config, fidelity={"budget": budget_i}, run_index=run_index
            )

            consumed_time += cost_step
            if DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME:
                time.sleep(cost_step * DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME)

            objective_i = 1 - eval["function_value"]  # maximizing in deephyper

            # Trial report is not support for MOO in Optuna
            if DEEPHYPER_BENCHMARK_MOO:
                continue

            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        objective = objective_i

    else:
        for budget_i in range(min_b, max_b + 1):
            eval = b.objective_function(
                config, fidelity={"budget": budget_i}, run_index=run_index
            )

            consumed_time += cost_step
            if DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME:
                time.sleep(cost_step * DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME)

            objective_i = 1 - eval["function_value"]  # maximizing in deephyper
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
        "objective": (objective, -consumed_time)
        if DEEPHYPER_BENCHMARK_MOO
        else objective,
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
