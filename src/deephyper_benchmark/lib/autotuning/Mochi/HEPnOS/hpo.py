import os
import time

import numpy as np
import pandas as pd
from deephyper.analysis.hpo import filter_failed_objectives
from deephyper.evaluator import RunningJob, profile
from deephyper.hpo import HpProblem
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
DATA_DIR = os.path.join(
    BUILD_DIR, "HEPnOS-Autotuning-analysis-main", "results", "theta"
)

DEEPHYPER_BENCHMARK_HEPNOS_MODEL = os.environ.get(
    "DEEPHYPER_BENCHMARK_HEPNOS_MODEL", "RAND"
)
DEEPHYPER_BENCHMARK_HEPNOS_SCALE = int(
    os.environ.get("DEEPHYPER_BENCHMARK_HEPNOS_SCALE", "4")
)
DEEPHYPER_BENCHMARK_HEPNOS_DISABLE_PEP = bool(
    os.environ.get("DEEPHYPER_BENCHMARK_HEPNOS_DISABLE_PEP", "0")
)
DEEPHYPER_BENCHMARK_HEPNOS_MORE_PARAMS = bool(
    os.environ.get("DEEPHYPER_BENCHMARK_HEPNOS_MORE_PARAMS", "0")
)

NUMERIC_VARIABLES = []
CATEGORICAL_VARIABLES = []

problem = HpProblem()


def add_parameter(name, domain, value_type, default_value, description=""):
    problem.add_hyperparameter(domain, name, default_value=default_value)
    if value_type in [int, float, bool]:
        NUMERIC_VARIABLES.append(name)
    else:
        CATEGORICAL_VARIABLES.append(name)


def create_problem():
    add_parameter(
        "busy_spin",
        [0, 1],
        int,
        0,
        "Whether Mercury should busy-spin instead of block",
    )
    add_parameter(
        "hepnos_progress_thread",
        [0, 1],
        int,
        0,
        "Whether to use a dedicated progress thread in HEPnOS",
    )
    add_parameter(
        "hepnos_num_rpc_threads",
        (0, 63),
        int,
        0,
        "Number of threads used for serving RPC requests",
    )
    add_parameter(
        "hepnos_num_event_databases",
        (1, 16),
        int,
        1,
        "Number of databases per process used to store events",
    )
    add_parameter(
        "hepnos_num_product_databases",
        (1, 16),
        int,
        1,
        "Number of databases per process used to store products",
    )
    add_parameter(
        "hepnos_num_providers",
        (1, 32),
        int,
        1,
        "Number of database providers per process",
    )
    add_parameter(
        "hepnos_pool_type",
        ["fifo", "fifo_wait"],
        str,
        "fifo_wait",
        "Thread-scheduling policity used by Argobots pools",
    )
    add_parameter(
        "hepnos_pes_per_node",
        [1, 2, 4, 8, 16, 32],
        int,
        2,
        "Number of HEPnOS processes per node",
    )
    add_parameter(
        "loader_progress_thread",
        [0, 1],
        int,
        0,
        "Whether to use a dedicated progress thread in the Dataloader",
    )
    add_parameter(
        "loader_batch_size",
        (1, 2048, "log-uniform"),
        int,
        512,
        "Size of the batches of events sent by the Dataloader to HEPnOS",
    )
    add_parameter(
        "loader_pes_per_node",
        [1, 2, 4, 8, 16],
        int,
        2,
        "Number of processes per node for the Dataloader",
    )
    if DEEPHYPER_BENCHMARK_HEPNOS_MORE_PARAMS:
        add_parameter(
            "loader_async",
            [0, 1],
            int,
            0,
            "Whether to use the HEPnOS AsyncEngine in the Dataloader",
        )
        add_parameter(
            "loader_async_threads",
            (1, 63, "log-uniform"),
            int,
            1,
            "Number of threads for the AsyncEngine to use",
        )

    if DEEPHYPER_BENCHMARK_HEPNOS_DISABLE_PEP:
        return

    add_parameter(
        "pep_progress_thread",
        [0, 1],
        int,
        0,
        "Whether to use a dedicated progress thread in the PEP step",
    )
    add_parameter(
        "pep_num_threads",
        (1, 31),
        int,
        4,
        "Number of threads used for processing in the PEP step",
    )
    add_parameter(
        "pep_ibatch_size",
        (8, 1024, "log-uniform"),
        int,
        128,
        "Batch size used when PEP processes are loading events from HEPnOS",
    )
    add_parameter(
        "pep_obatch_size",
        (8, 1024, "log-uniform"),
        int,
        128,
        "Batch size used when PEP processes are exchanging events among themselves",
    )
    add_parameter(
        "pep_pes_per_node",
        [1, 2, 4, 8, 16, 32],
        int,
        8,
        "Number of processes per node for the PEP step",
    )

    if DEEPHYPER_BENCHMARK_HEPNOS_MORE_PARAMS:
        add_parameter(
            "pep_no_preloading",
            [0, 1],
            int,
            False,
            "Whether to disable product-preloading in PEP",
        )
        add_parameter(
            "pep_no_rdma",
            [0, 1],
            int,
            0,
            "Whether to disable RDMA in PEP",
        )


create_problem()

CSV_SUFFIX = f"{str(DEEPHYPER_BENCHMARK_HEPNOS_DISABLE_PEP).lower()}-{str(DEEPHYPER_BENCHMARK_HEPNOS_MORE_PARAMS).lower()}"
CSV_SUFFIX = f"{DEEPHYPER_BENCHMARK_HEPNOS_MODEL}-{DEEPHYPER_BENCHMARK_HEPNOS_SCALE}-{CSV_SUFFIX}"


def load_data():
    dataframes, dataframes_with_failures = [], []
    for i in range(1, 6):
        csv_path = os.path.join(
            DATA_DIR,
            f"exp-{CSV_SUFFIX}-{i}.csv",
        )
        df = pd.read_csv(csv_path, index_col=None, header=0)
        df, df_with_failures = filter_failed_objectives(df)
        dataframes.append(df)
        dataframes_with_failures.append(df_with_failures)

    dataframes = pd.concat(dataframes, axis=0, ignore_index=True)
    dataframes_with_failures = pd.concat(
        dataframes_with_failures, axis=0, ignore_index=True
    )

    # The objective is -log(time) initialy so we convert it back to time
    dataframes["objective"] = dataframes["objective"] = np.exp(
        -dataframes["objective"].values.astype(np.float32)
    )
    dataframes_with_failures["objective"] = -1

    dataframes = pd.concat(
        [dataframes, dataframes_with_failures], axis=0, ignore_index=True
    )

    return dataframes


def create_model():
    df = load_data()
    names = problem.hyperparameter_names

    X = df.loc[:, names]
    y = df["objective"].values

    transformer = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_VARIABLES,
            ),
            ("passthrough", FunctionTransformer(), NUMERIC_VARIABLES),
        ]
    )

    model = KNeighborsRegressor(n_neighbors=1)

    pipeline = Pipeline([("transformer", transformer), ("model", model)])
    pipeline.fit(X, y)

    return pipeline


model = create_model()


@profile
def run(job: RunningJob) -> dict:
    config = job.parameters.copy()

    df = pd.DataFrame([config])
    y_pred = -model.predict(df)[0]

    if y_pred > 0:
        y_pred = "F"

    return {"objective": y_pred}


if __name__ == "__main__":
    print(problem)
    df = load_data()
    print(df)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
