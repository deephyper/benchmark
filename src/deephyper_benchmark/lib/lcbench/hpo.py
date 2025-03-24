import os

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import profile, RunningJob
from deephyper.hpo import HpProblem

dataset = os.environ.get("DEEPHYPER_BENCHMARK_LCBENCH_DATASET", "APSFailure")

data_path = os.path.join(DIR, "../build/LCBench/data/data_2k.json")

from api import Benchmark as LCBenchBenchmark
bench = LCBenchBenchmark(data_path, cache=True, cache_dir=os.path.join(DIR, "../build/LCBench/data/"))

num_elements = 2000

problem = HpProblem()
problem.add_hyperparameter((0, num_elements - 1), "index")


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:

    
    # otherwise failure
    idx = job.parameters["index"]
    y_valid = bench.query(dataset, "Train/val_cross_entropy", idx)[1:51]
    y_test  = bench.query(dataset, "Train/test_cross_entropy", idx)[1:51]

    min_b, max_b = 1, 50 

    objective_val = -y_valid[-1]
    objective_test = -y_test[-1]

    other_metadata = {}

    if optuna_trial:

        for budget_i in range(min_b, max_b + 1):
            objective_i = -y_valid[budget_i-1] # maximizing in deephyper
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        objective = objective_i

    else:

        for budget_i in range(min_b, max_b + 1):
            objective_i = -y_valid[budget_i-1]  # maximizing in deephyper
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
