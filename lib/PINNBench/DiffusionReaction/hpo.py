import os

import numpy as np
import torch
from deephyper.evaluator import RunningJob, profile
from deephyper.problem import HpProblem
from deephyper.stopper.integration.deepxde import DeepXDEStopperCallback
from deepxde.callbacks import EarlyStopping
from fvcore.nn import FlopCountAnalysis
from pdebench.models.pinn.train import run_training

from deephyper_benchmark.integration.torch import count_params
from deephyper_benchmark.utils.json_utils import array_to_json

from .model import FNN, ACTIVATIONS

DIR = os.path.dirname(os.path.abspath(__file__))
DEEPHYPER_BENCHMARK_MOO = bool(int(os.environ.get("DEEPHYPER_BENCHMARK_MOO", 0)))

# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=10)
problem.add_hyperparameter((5, 100), "num_neurons", default_value=10)
problem.add_hyperparameter((100, 1000), "epochs", default_value=100)
problem.add_hyperparameter(
    list(ACTIVATIONS.keys()),
    "activation",
    default_value="relu",
)
problem.add_hyperparameter([True, False], "skip_co", default_value=False)
problem.add_hyperparameter((0.0, 0.5), "dropout", default_value=0)

# Regularization hyperparameters
problem.add_hyperparameter([True, False], "batch_norm", default_value=False)
problem.add_hyperparameter(["None", "l2"], "regularization", default_value="None")
problem.add_hyperparameter(
    (1e-5, 1.0, "log-uniform"), "weight_decay", default_value=0.01
)
problem.add_hyperparameter(
    ["Glorot normal", "Glorot uniform", "He normal", "He uniform"],
    "kernel_initializer",
    default_value="Glorot normal",
)

# Optimization hyperparameters
problem.add_hyperparameter(["None", "step"], "decay", default_value="None")
problem.add_hyperparameter((1, 100), "decay_step_size", default_value=5)
problem.add_hyperparameter((1e-5, 1.0, "log-uniform"), "decay_gamma", default_value=0.1)

problem.add_hyperparameter(
    ["adam", "sgd", "rmsprop", "adamw"], "optimizer", default_value="adam"
)
problem.add_hyperparameter(
    (1e-5, 1e-1, "log-uniform"), "learning_rate", default_value=0.01
)

# Loss weights is tuned only if MOO is activated.
if DEEPHYPER_BENCHMARK_MOO:
    problem.add_hyperparameter((0.1, 0.9), "loss_weights", default_value=0.5)


@profile
def run(job: RunningJob) -> dict:
    config = job.parameters.copy()
    dataset = "2D_diff-react_NA_NA"

    for k, v in config.items():
        if v == "None":
            config[k] = None

    if "loss_weights" not in config:
        lw = 0.5
    else:
        lw = config["loss_weights"]
    config["loss_weights"] = np.array([lw, lw, 1 - lw, 1 - lw, 1 - lw, 1 - lw])

    # https://github.com/lululxvi/deepxde/blob/master/deepxde/optimizers/pytorch/optimizers.py
    if config["decay"] == "step":
        config["decay"] = ("step", config["decay_step_size"], config["decay_gamma"])

    # To avoid error: https://github.com/lululxvi/deepxde/blob/master/deepxde/optimizers/pytorch/optimizers.py#L45C4-L45C4
    if config["optimizer"] == "adamw":
        config["regularization"] = "l2"

    stopper_callback = DeepXDEStopperCallback(job)

    error_type = None
    try:
        (
            val_loss,
            test_loss,
            losshistory,
            model,
            duration_batch_inference,
        ) = run_training(
            net_class=FNN,
            scenario="diff-react",
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            model_update=500,
            root_path=os.path.join(
                DIR, "../build/PDEBench-DH/pdebench/data/" + dataset
            ),
            flnm=dataset + ".h5",
            config=config,
            seed="0000",
            callbacks=stopper_callback,
        )
    except torch.cuda.OutOfMemoryError:
        error_type = "F_OOM"

    # Managing failures to learn memory constraints
    if error_type is not None:
        metadata = {
            "num_parameters": None,
            "num_parameters_train": None,
            "val_loss": None,
            "test_rmse": None,
            "budget": None,
            "stopped": None,
            "lc_train_loss": None,
            "lc_val_loss": None,
            "flops": None,
            "duration_batch_inference": None,
        }

        if DEEPHYPER_BENCHMARK_MOO:
            objective = [error_type for _ in range(3)]
        else:
            objective = error_type

        return {"objective": objective, "metadata": metadata}

    param_count = count_params(model)
    flops = FlopCountAnalysis(model, inputs=(torch.randn(1, 3))).total()

    train_ls = np.array(losshistory.loss_train).sum(axis=1)
    val_ls = np.array(losshistory.loss_test).sum(axis=1)
    steps = np.array(losshistory.steps)
    lc_train_X = np.stack([steps, train_ls], axis=1)
    lc_val_X = np.stack([steps, val_ls], axis=1)
    lc_train_X_json = array_to_json(lc_train_X)
    lc_val_X_json = array_to_json(lc_val_X)

    if DEEPHYPER_BENCHMARK_MOO:
        print("Optimizing multiple objectives...")

        if np.isnan(val_loss[:2]).any() or np.isinf(val_loss[:2]).any():
            objective_0 = "F"
        else:
            objective_0 = -(val_loss[:2] / config["loss_weights"][:2]).sum()

        if np.isnan(val_loss[2:]).any() or np.isinf(val_loss[2:]).any():
            objective_1 = "F"
        else:
            objective_1 = -(val_loss[2:] / config["loss_weights"][2:]).sum()

        objective = [
            objective_0,
            objective_1,
            -flops,
        ]
    else:
        objective = (
            "F"
            if np.isnan(val_loss).any() or np.isinf(val_loss).any()
            else -(val_loss / config["loss_weights"]).sum()
        )
    metadata = {
        "num_parameters": param_count["num_parameters"],
        "num_parameters_train": param_count["num_parameters_train"],
        "val_loss": array_to_json(val_loss),  # array of 4 elements
        "test_rmse": float(test_loss[0]),
        "budget": stopper_callback.budget,
        "stopped": stopper_callback.stopped,
        "lc_train_loss": lc_train_X_json,
        "lc_val_loss": lc_val_X_json,
        "flops": flops,
        "duration_batch_inference": duration_batch_inference,  # add the inference time in seconds
    }

    return {"objective": objective, "metadata": metadata}


def evaluate(config):
    """
    Evaluate an hyperparameter configuration
    on training/validation and testing data.
    """
    callbacks = EarlyStopping(patience=100_000)
    DIR = os.path.dirname(os.path.abspath(__file__))
    dataset = os.environ.get("DEEPHYPER_BENCHMARK_DATASET")

    val_loss, test_loss, losshistory, model = run_training(
        net_class=FNN,
        scenario="diff-react",
        epochs=config["epochs"],
        learning_rate=config["lr"],
        model_update=1,
        root_path=os.path.join(DIR, "../build/PDEBench-DH/pdebench/data/" + dataset),
        flnm="2D_diff-react_NA_NA.h5",
        config=config,
        seed="0000",
        callbacks=callbacks,
    )
    return val_loss, test_loss, losshistory
