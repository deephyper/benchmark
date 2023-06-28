import copy
import traceback

from deephyper.evaluator import profile
from deephyper.problem import HpProblem
from deephyper.stopper.integration import TFKerasStopperCallback

from .model import run_pipeline


problem = HpProblem()

# Model hyperparameters
ACTIVATIONS = [
    "elu",
    "gelu",
    "hard_sigmoid",
    "linear",
    "relu",
    "selu",
    "sigmoid",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]
default_dense = [1000, 1000, 1000]
default_dense_feature_layers = [1000, 1000, 1000]

for i in range(len(default_dense)):

    problem.add_hyperparameter(
        (10, 1024, "log-uniform"),
        f"dense_{i}",
        default_value=default_dense[i],
    )

    problem.add_hyperparameter(
        (10, 1024, "log-uniform"),
        f"dense_feature_layers_{i}",
        default_value=default_dense_feature_layers[i],
    )

problem.add_hyperparameter(ACTIVATIONS, "activation", default_value="relu")

# Optimization hyperparameters
problem.add_hyperparameter(
    [
        "sgd",
        "rmsprop",
        "adagrad",
        "adadelta",
        "adam",
    ],
    "optimizer",
    default_value="sgd",
)

problem.add_hyperparameter((0, 0.5), "dropout", default_value=0.0)
problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=32)

problem.add_hyperparameter(
    (1e-5, 1e-2, "log-uniform"), "learning_rate", default_value=0.001
)
problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "base_lr", default_value=0.001)
problem.add_hyperparameter([True, False], "residual", default_value=False)

problem.add_hyperparameter([True, False], "early_stopping", default_value=False)
problem.add_hyperparameter((5, 20), "early_stopping_patience", default_value=5)

problem.add_hyperparameter([True, False], "reduce_lr", default_value=False)
problem.add_hyperparameter((0.1, 1.0), "reduce_lr_factor", default_value=0.5)
problem.add_hyperparameter((5, 20), "reduce_lr_patience", default_value=5)

problem.add_hyperparameter([True, False], "warmup_lr", default_value=False)
problem.add_hyperparameter([True, False], "batch_normalization", default_value=False)

problem.add_hyperparameter(
    ["mse", "mae", "logcosh", "mape", "msle", "huber"], "loss", default_value="mse"
)

problem.add_hyperparameter(["std", "minmax", "maxabs"], "scaling", default_value="std")


def remap_hyperparameters(config: dict):
    """Transform input configurations of hyperparameters to the format accepted by the candle benchmark."""
    dense = []
    dense_feature_layers = []
    for i in range(len(default_dense)):

        key = f"dense_{i}"
        dense.append(config.pop(key))

        key = f"dense_feature_layers_{i}"
        dense_feature_layers.append(config.pop(key))

    config["dense"] = dense
    config["dense_feature_layers"] = dense_feature_layers


@profile
def run(job, optuna_trial=None):

    config = copy.deepcopy(job.parameters)

    params = {
        "epochs": 50,
        "timeout": 60 * 30,  # 30 minutes per model
        "verbose": False,
    }
    if len(config) > 0:
        remap_hyperparameters(config)
        params.update(config)

    if optuna_trial is None:
        stopper_callback = TFKerasStopperCallback(
            job, monitor="val_r2", mode="max" 
        )
    else:

        from deephyper_benchmark.integration.optuna import KerasPruningCallback

        stopper_callback = KerasPruningCallback(optuna_trial, "val_r2")

    try:
        score = run_pipeline(params, mode="valid", stopper_callback=stopper_callback)
    except Exception as e:
        print(traceback.format_exc())
        score = {"objective": "F"}
        keys = "m:num_parameters,m:num_parameters_train,m:budget,m:stopped,m:train_mse,m:train_mae,m:train_r2,m:train_corr,m:valid_mse,m:valid_mae,m:valid_r2,m:valid_corr,m:test_mse,m:test_mae,m:test_r2,m:test_corr,m:lc_train_mse,m:lc_valid_mse,m:lc_train_mae,m:lc_valid_mae,m:lc_train_r2,m:lc_valid_r2"
        metadata = {k.strip("m:"): None for k in keys.split(",")}
        score["metadata"] = metadata
        
    return score


def evaluate(config):
    """Evaluate an hyperparameter configuration on training/validation and testing data."""

    params = {
        "epochs": 100,
        "timeout": 60 * 60,
        "verbose": True,
    }  # 60 minutes per model
    remap_hyperparameters(config)
    params.update(config)
    run_pipeline(params, mode="test")
