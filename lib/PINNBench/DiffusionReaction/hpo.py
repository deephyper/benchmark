import os
import torch
import numpy as np
from pdebench.models.pinn.train import *
from .model import FNN
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import profile, RunningJob
from deephyper.stopper.integration import DeepXDEStopperCallback
from deephyper.stopper import LCModelStopper
from deephyper_benchmark.utils.json_utils import array_to_json
from fvcore.nn import FlopCountAnalysis
from deephyper_benchmark.integration.torch import count_params
from deepxde.callbacks import EarlyStopping


@profile
def run(job: RunningJob) -> dict:
    config = job.parameters
    dataset = '2D_diff-react_NA_NA'
    DEEPHYPER_BENCHMARK_MOO = bool(int(os.environ.get("DEEPHYPER_BENCHMARK_MOO", 0)))
    DIR = os.path.dirname(os.path.abspath(__file__))
    stopper_callback = DeepXDEStopperCallback(job)

    val_loss, test_loss, losshistory, model, duration_batch_inference = run_training(
        net_class=FNN,
        scenario="diff-react",
        epochs=config["epochs"],
        learning_rate=config["lr"],
        model_update=500,
        root_path=os.path.join(DIR, "../build/PDEBench-DH/pdebench/data/" + dataset),
        flnm=dataset + ".h5",
        config=config,
        seed="0000",
        callbacks=stopper_callback,
    )
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
        objective = [
            -sum(val_loss[:2]),
            -sum(val_loss[2:]),
            -param_count["num_parameters_train"],
            -duration_batch_inference,
            -flops
        ]
    else:
        objective = -sum(val_loss)
    metadata = {
        "num_parameters": param_count["num_parameters"],
        "num_parameters_train": param_count["num_parameters_train"],
        "val_loss": val_loss,
        "test_rmse": test_loss[0],
        "budget": stopper_callback.budget,
        "stopped": job.stopped(),
        "lc_train_loss": lc_train_X_json,
        "lc_val_loss": lc_val_X_json,
        "flops": flops,
        "duration_batch_inference": duration_batch_inference,  # add the inference time in seconds
    }

    return {"objective": objective, "metadata": metadata}


# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=5)
problem.add_hyperparameter((1e-5, 1e-2), "lr", default_value=0.01)
problem.add_hyperparameter((5, 30), "num_neurons", default_value=5)
problem.add_hyperparameter((100, 1000), "epochs", default_value=100)
problem.add_hyperparameter(
    ["relu", "swish", "tanh", "elu", "selu", "sigmoid"],
    "activation",
    default_value="tanh",
)
problem.add_hyperparameter(["True", "False"], "skip_co", default_value="False")
problem.add_hyperparameter((0, 1.0), "dropout_rate", default_value=0)
problem.add_hyperparameter(
    ["adam", "sgd", "rmsprop", "adamw"], "optimizer", default_value="adam"
)
problem.add_hyperparameter((0, 0.1), "weight_decay", default_value=0)
problem.add_hyperparameter(
    ["Glorot normal", "Glorot uniform", "He normal", "He uniform", "zeros"],
    "initialization",
    default_value="Glorot normal",
)
problem.add_hyperparameter((0.1, 0.9), 'loss_weights', default_value=0.5)


def evaluate(config):
    """
    Evaluate an hyperparameter configuration
    on training/validation and testing data.
    """
    from deepxde.callbacks import EarlyStopping

    callbacks = EarlyStopping(patience=100000)
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
