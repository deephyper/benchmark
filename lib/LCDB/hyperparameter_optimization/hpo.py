import gzip
import os
import time

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import RunningJob, profile
from deephyper.problem import HpProblem
from lcdb.analysis import read_csv_results
from lcdb.analysis.json import QueryAnchorValues, QueryMetricValuesFromAnchors
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix
from lcdb.utils import import_attr_from_module

# DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME = bool(
#     int(os.environ.get("DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME", 0))
# )
# DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME = float(
#     os.environ.get("DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME", 1.0)
# )

DEEPHYPER_BENCHMARK_WORKFLOW = "lcdb.workflow.sklearn.KNNWorkflow"
DEEPHYPER_BENCHMARK_CSV = os.environ.get(
    "DEEPHYPER_BENCHMARK_CSV",
    "/Users/romainegele/Documents/Research/LCDB/lcdb/publications/2023-neurips/experiments/alcf/polaris/knn/output/lcdb.workflow.sklearn.KNNWorkflow/3/42-42-42/results.csv.gz",
)


# WorkflowClass = import_attr_from_module(DEEPHYPER_BENCHMARK_WORKFLOW)
# config_space = WorkflowClass.config_space()
# problem = HpProblem(config_space=config_space)
problem = HpProblem()

with gzip.GzipFile(DEEPHYPER_BENCHMARK_CSV, "rb") as f:
    r_df, r_df_failed = read_csv_results(f)
DEEPHYPER_BENCHMARK_DATA = r_df

problem.add_hyperparameter((0, len(DEEPHYPER_BENCHMARK_DATA) - 1), "eval_id")
query_anchor_values = QueryAnchorValues()
query_confusion_matrix_values_on_val = QueryMetricValuesFromAnchors(
    "confusion_matrix", split_name="val"
)


def query_balanced_error_rate_values_on_val(x):
    return list(
        map(
            lambda x: 1 - balanced_accuracy_from_confusion_matrix(x),
            query_confusion_matrix_values_on_val(x),
        )
    )


query_confusion_matrix_values_on_test = QueryMetricValuesFromAnchors(
    "confusion_matrix", split_name="test"
)


def query_balanced_error_rate_values_on_test(x):
    return list(
        map(
            lambda x: 1 - balanced_accuracy_from_confusion_matrix(x),
            query_confusion_matrix_values_on_test(x),
        )
    )


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:
    # otherwise failure
    eval_id = job.parameters["eval_id"]

    source = DEEPHYPER_BENCHMARK_DATA.iloc[eval_id]["m:json"]
    anchor_values = query_anchor_values(source)
    objective_val_values = query_balanced_error_rate_values_on_val(source)
    objective_test_values = query_balanced_error_rate_values_on_test(source)

    anchor_values = anchor_values[: len(objective_val_values)]

    min_b, max_b = anchor_values[0], anchor_values[-1]

    other_metadata = {}

    consumed_time = 0  # the cost here corresponds to time

    if optuna_trial:
        for i, budget_i in enumerate(anchor_values):
            objective_i = -objective_val_values[i]

            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        objective = objective_i

    else:
        for i, budget_i in enumerate(anchor_values):
            objective_i = -objective_val_values[i]

            job.record(budget_i, 1+objective_i)
            if job.stopped():
                break

        objective = job.objective

        if hasattr(job, "stopper") and hasattr(job.stopper, "infos_stopped"):
            other_metadata["infos_stopped"] = job.stopper.infos_stopped

    metadata = {
        "budget": budget_i,
        "stopped": budget_i < max_b,
        "objective_test": -objective_test_values[-1],
        "objective_val": -objective_val_values[-1],
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
