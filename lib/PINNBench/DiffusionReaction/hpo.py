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


@profile
def run(job: RunningJob) -> dict:
    config = job.parameters
    dataset = os.environ.get('DEEPHYPER_BENCHMARK_DATASET')

    stopper_callback = DeepXDEStopperCallback(job)

    val_loss, test_loss, losshistory, model = run_training(
        net_class=FNN,
        scenario="diff-react",
        epochs=config["epochs"],
        learning_rate=config["lr"],
        model_update=500,
        root_path="./DiffusionReaction/build/PDEBench-DH/pdebench/data/" + dataset,
        flnm=dataset + ".h5",
        config=config,
        seed="0000",
        callbacks=stopper_callback,
    )
    param_count = count_params(model)
    flops = FlopCountAnalysis(model, inputs=(torch.randn(1, 3))).total()

    train_ls = np.array(losshistory.loss_train).sum(axis=1)
    val_ls = np.array(losshistory.loss_test).sum(axis=1),
    epoch_seq = np.arange(train_ls.shape[0])
    lc_train_X = np.stack(epoch_seq, train_ls, axis=1)
    lc_val_X = np.stack(epoch_seq, val_ls, axis=1)
    lc_train_X_json = array_to_json(lc_train_X)
    lc_val_X_json = array_to_json(lc_val_X)

    objective = -val_loss
    metadata = {
        "num_hyperparameters": len(job.parameters),
        "num_parameters":param_count['num_parameters'],
        "num_parameters_train":param_count["num_parameters_train"],
        "val_loss": val_loss,
        "test_loss": test_loss,
        "budget": stopper_callback.budget,
        "stopped":job.stopped,
        "lc_train_X": lc_train_X_json,
        "lc_val_X": lc_val_X_json,
        "FLOPS": flops

    }
    return {"objective": objective, "metadata": metadata}


# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=5)
problem.add_hyperparameter((1e-5, 1e-2), "lr", default_value=0.01)
problem.add_hyperparameter((5, 50), "num_neurons", default_value=5)
problem.add_hyperparameter((1, 200), "epochs", default_value=1)
problem.add_hyperparameter(
    ["relu", "swish", "tanh", "elu", "selu", "sigmoid"],
    "activation",
    default_value="tanh",
)


def evaluate(config):
    """
    Evaluate an hyperparameter configuration
    on training/validation and testing data.
    """
    from deepxde.callbacks import EarlyStopping

    callbacks = EarlyStopping(patience=100000)

    val_loss, test_loss, losshistory = run_training(
        net_class=FNN,
        scenario="diff-react",
        epochs=config["epochs"],
        learning_rate=config["lr"],
        model_update=500,
        root_path="~/Downloads/2D/diffusion-reaction/",
        flnm="2D_diff-react_NA_NA.h5",
        config=config,
        seed="0000",
        callbacks=callbacks,
    )
    return val_loss, test_loss, losshistory


# if __name__ == "__main__":
#     default_config = problem.default_configuration
#     # result = run(RunningJob(parameters=default_config))
#     # print(f"{result=}")

#     # stopper = SuccessiveHalvingStopper(min_steps=1, max_steps=200)
#     # search = CBO(
#     #        problem, run, initial_points=[problem.default_configuration], stopper=stopper
#     #)
#     # results = search.search(max_evals=100)
#     # best_config = {'epochs': 10000, 'lr':1e-3, 'num_layers':6, 'num_neurons':40, 'activation': 'tanh'}
#     # eval_val_loss, eval_test_loss, losshistory = evaluate(best_config)
#     # print(eval_val_loss, eval_test_loss)

#     # import pickle
#     # with open('./history.pkl', 'wb') as f:
#     #     pickle.dump(losshistory, f)
