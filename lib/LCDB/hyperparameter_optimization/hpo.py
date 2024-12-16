import gzip
import os
import time

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

from deephyper.evaluator import RunningJob, profile
from deephyper.hpo import HpProblem
from lcdb.analysis import read_csv_results
from lcdb.analysis.json import (
    QueryAnchorValues,
    QueryMetricValuesFromAnchors,
    QueryEpochValues,
    QueryMetricValuesFromEpochs,
)
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix
from lcdb.utils import import_attr_from_module

# DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME = bool(
#     int(os.environ.get("DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME", 0))
# )
# DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME = float(
#     os.environ.get("DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME", 1.0)
# )

# DEEPHYPER_BENCHMARK_FIDELITY = os.environ.get("DEEPHYPER_BENCHMARK_FIDELITY", "anchor")
# DEEPHYPER_BENCHMARK_WORKFLOW = "lcdb.workflow.sklearn.KNNWorkflow"
# DEEPHYPER_BENCHMARK_CSV = os.environ.get(
#     "DEEPHYPER_BENCHMARK_CSV",
#     "/Users/romainegele/Documents/Research/LCDB/lcdb/publications/2023-neurips/experiments/alcf/polaris/knn/output/lcdb.workflow.sklearn.KNNWorkflow/3/42-42-42/results.csv.gz",
# )
DEEPHYPER_BENCHMARK_MAX_FIDELITY = int(
    os.environ.get("DEEPHYPER_BENCHMARK_MAX_FIDELITY", 100)
)
DEEPHYPER_BENCHMARK_FIDELITY = os.environ.get("DEEPHYPER_BENCHMARK_FIDELITY", "epoch")
DEEPHYPER_BENCHMARK_CSV = os.environ.get(
    "DEEPHYPER_BENCHMARK_CSV",
    "/Users/romainegele/Documents/Research/LCDB/lcdb/publications/2023-neurips/experiments/alcf/polaris/densenn/output/lcdb.workflow.keras.DenseNNWorkflow/3/42-42-42/results.csv.gz",
)


# WorkflowClass = import_attr_from_module(DEEPHYPER_BENCHMARK_WORKFLOW)
# config_space = WorkflowClass.config_space()
# problem = HpProblem(config_space=config_space)
problem = HpProblem()

with gzip.GzipFile(DEEPHYPER_BENCHMARK_CSV, "rb") as f:
    r_df, r_df_failed = read_csv_results(f)
DEEPHYPER_BENCHMARK_DATA = r_df

if DEEPHYPER_BENCHMARK_FIDELITY == "anchor":
    query_anchor_values = QueryAnchorValues()
    query_confusion_matrix_values_on_val = QueryMetricValuesFromAnchors(
        "confusion_matrix", split_name="val"
    )
    query_confusion_matrix_values_on_test = QueryMetricValuesFromAnchors(
        "confusion_matrix", split_name="test"
    )
elif DEEPHYPER_BENCHMARK_FIDELITY == "epoch":
    # Load the learning "epoch"-curve at last epoch
    query_anchor_values_ = QueryEpochValues()
    query_confusion_matrix_values_on_val_ = QueryMetricValuesFromEpochs(
        "confusion_matrix", split_name="val"
    )
    query_confusion_matrix_values_on_test_ = QueryMetricValuesFromEpochs(
        "confusion_matrix", split_name="test"
    )
    query_anchor_values = lambda x: query_anchor_values_(x)[-1]
    query_confusion_matrix_values_on_val = (
        lambda x: query_confusion_matrix_values_on_val_(x)[-1]
    )
    query_confusion_matrix_values_on_test = (
        lambda x: query_confusion_matrix_values_on_test_(x)[-1]
    )

    source = DEEPHYPER_BENCHMARK_DATA["m:json"]
    values = source.apply(lambda x: query_confusion_matrix_values_on_val(x)).to_list()
    mask = [len(x) > 0 for x in values]
    DEEPHYPER_BENCHMARK_DATA = DEEPHYPER_BENCHMARK_DATA[mask]

    source = DEEPHYPER_BENCHMARK_DATA["m:json"]
    values = source.apply(lambda x: query_confusion_matrix_values_on_test(x)).to_list()
    mask = [len(x) > 0 for x in values]
    DEEPHYPER_BENCHMARK_DATA = DEEPHYPER_BENCHMARK_DATA[mask]
else:
    raise ValueError(f"Unknown fidelity: {DEEPHYPER_BENCHMARK_FIDELITY}")

problem.add_hyperparameter((0, len(DEEPHYPER_BENCHMARK_DATA) - 1), "eval_id")


# TODO: To test LCPFN with accuracy metric
def query_balanced_accuracy_values_on_val(x):
    cm = query_confusion_matrix_values_on_val(x)
    return list(map(balanced_accuracy_from_confusion_matrix, cm))


def query_coefficient_of_determiniation_values_on_val(x):
    cm = query_confusion_matrix_values_on_val(x)
    num_classes = len(cm[0])
    balanced_error_rate_baseline = 1 - 1 / num_classes
    return list(
        map(
            lambda x: 1
            - (1 - balanced_accuracy_from_confusion_matrix(x))
            / balanced_error_rate_baseline,
            cm,
        )
    )


def query_coefficient_of_determiniation_values_on_test(x):
    cm = query_confusion_matrix_values_on_test(x)
    num_classes = len(cm[0])
    balanced_error_rate_baseline = 1 - 1 / num_classes
    return list(
        map(
            lambda x: 1
            - (1 - balanced_accuracy_from_confusion_matrix(x))
            / balanced_error_rate_baseline,
            cm,
        )
    )


@profile
def run(job: RunningJob, optuna_trial=None) -> dict:
    # otherwise failure
    eval_id = job.parameters["eval_id"]

    source = DEEPHYPER_BENCHMARK_DATA.iloc[eval_id]["m:json"]
    anchor_values = query_anchor_values(source)

    # TODO: to test with LCPFN
    objective_values = query_balanced_accuracy_values_on_val(source)

    objective_val_values = query_coefficient_of_determiniation_values_on_val(source)
    objective_test_values = query_coefficient_of_determiniation_values_on_test(source)
    length = min(len(objective_val_values), len(objective_test_values))
    anchor_values = anchor_values[:length]
    objective_values = objective_values[:length]
    objective_val_values = objective_val_values[:length]
    objective_test_values = objective_test_values[:length]

    objective_values = np.array(objective_values)
    objective_val_values = np.array(objective_val_values)
    objective_test_values = np.array(objective_test_values)
    anchor_values = np.array(anchor_values)

    mask = anchor_values <= DEEPHYPER_BENCHMARK_MAX_FIDELITY
    anchor_values = anchor_values[mask].tolist()
    objective_values = objective_values[mask].tolist()
    objective_val_values = objective_val_values[mask].tolist()
    objective_test_values = objective_test_values[mask].tolist()

    if anchor_values[-1] <= DEEPHYPER_BENCHMARK_MAX_FIDELITY:
        anchor_values.append(DEEPHYPER_BENCHMARK_MAX_FIDELITY)
        objective_values.append(objective_values[-1])
        objective_val_values.append(objective_val_values[-1])
        objective_test_values.append(objective_test_values[-1])

    min_b, max_b = anchor_values[0], anchor_values[-1]

    other_metadata = {}

    consumed_time = 0  # the cost here corresponds to time

    if optuna_trial:
        for i, budget_i in enumerate(anchor_values):
            objective_i = objective_val_values[i]

            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        objective = objective_i

    else:
        for i, budget_i in enumerate(anchor_values):
            # objective_i = objective_val_values[i]
            # TODO: to test with LCPFN
            objective_i = objective_values[i]

            job.record(budget_i, objective_i)
            if job.stopped():
                break

        objective = job.objective

        if hasattr(job, "stopper") and hasattr(job.stopper, "infos_stopped"):
            other_metadata["infos_stopped"] = job.stopper.infos_stopped

    metadata = {
        "budget": budget_i,
        "stopped": budget_i < max_b,
        "objective_test": objective_test_values[-1],
        "objective_val": objective_val_values[-1],
    }
    metadata.update(other_metadata)

    return {
        "objective": objective,
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(problem)
    # default_config = problem.default_configuration
    for eval_id in range(10):
        default_config = dict(eval_id=eval_id)
        print(f"{default_config=}")
        result = run(RunningJob(parameters=default_config))
        print(f"{result=}")
        print()
