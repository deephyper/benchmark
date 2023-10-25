import os

from ConfigSpace import ConfigurationSpace
from deephyper.evaluator import profile, RunningJob
from deephyper.problem import HpProblem

from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

DEEPHYPER_BENCHMARK_INSTANCE = os.environ.get("DEEPHYPER_BENCHMARK_INSTANCE", "3945")


bench = benchmark_set.BenchmarkSet("lcbench")
bench.set_instance(DEEPHYPER_BENCHMARK_INSTANCE)

config_space = ConfigurationSpace()
config_space.add_hyperparameters(
    [
        hp
        for hp in bench.config_space.get_hyperparameters()
        if not (hp.name in ["OpenML_task_id", "epoch"])
    ]
)
problem = HpProblem(config_space=config_space)


@profile()
def run(job: RunningJob) -> dict:
    config = job.parameters.copy()
    config["OpenML_task_id"] = DEEPHYPER_BENCHMARK_INSTANCE

    # Min/Max budget (here epochs)
    min_b, max_b = 2, 51

    # Run the benchmark (batch for better performance)
    def update_config(config, budget):
        config = config.copy()
        config["epoch"] = budget
        return config

    outputs = [
        bench.objective_function(update_config(config, budget=b))
        for b in range(min_b, max_b + 1)
    ]

    for i, out_i in enumerate(outputs):
        out_i = out_i[0]
        budget_i = i + 1
        objective_i = -out_i["val_cross_entropy"]

        job.record(budget_i, objective_i)
        if job.stopped():
            break

    return {
        "objective": objective_i,
        "metadata": {
            "budget": budget_i,
            "stopped": budget_i < len(outputs),
            "test_balanced_accuracy": outputs[-1][0]["test_balanced_accuracy"],
            "val_balanced_accuracy": outputs[-1][0]["val_balanced_accuracy"],
            "test_cross_entropy": outputs[-1][0]["test_cross_entropy"],
            "val_cross_entropy": outputs[-1][0]["val_cross_entropy"],
            "time": outputs[-1][0]["time"],
        },
    }


if __name__ == "__main__":
    print(problem)
    config = bench.config_space.sample_configuration(1).get_dictionary()
    config.pop("OpenML_task_id")
    print(config)
    result = run(RunningJob(id=0, parameters=config))
    print(result)
