import os
import numpy as np
from pdebench.models.pinn.train import *
from .model import FNN
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import profile, RunningJob
from deephyper.stopper.integration import DeepXDEStopperCallback
from deephyper.stopper import LCModelStopper


@profile
def run(job: RunningJob) -> dict:
    config = job.parameters
    dataset = os.environ.get('DEEPHYPER_BENCHMARK_DATASET')

    stopper_callback = DeepXDEStopperCallback(job)

    val_loss, test_loss, losshistory = run_training(
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
    hist = {
        "train": np.array(losshistory.loss_train).sum(axis=1),
        "val": np.array(losshistory.loss_test).sum(axis=1),
    }
    objective = -val_loss
    metadata = {
        "num_hyperparameters": len(job.parameters),
        "val_loss": val_loss,
        "test_loss": test_loss,
        "budget": stopper_callback.budget,
        "learning_curve": hist,
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
