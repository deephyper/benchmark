import torch
from deephyper.evaluator import profile
from deephyper.problem import HpProblem

from .model import build_and_train_model

# now we define the hyperparameters with which we aim to get the best configuration for solving the problem
problem = HpProblem()
# this is for dimension 2 darcy flow, below we list the hyperparameters and where they are being used
# 1. n_modes_height and n_modes_width are the number of modes to keep in fourier layer along each dimension, if we are working with 3 dimensions, we have height, width, and length. It is called when you are defining the model e.g model =TFNO(n_modes=...),The dimensionality of the TFNO is inferred from ``len(n_modes)``
problem.add_hyperparameter((2, 19), "n_modes_height", default_value=16)
problem.add_hyperparameter((2, 19), "n_modes_width", default_value=16)
# 2. hidden_channels: width of the FNO (i.e. number of channels). they go into the model definition
problem.add_hyperparameter((1, 128), "hidden_channels", default_value=32)
# 3. lifting_channels: number of hidden channels of the lifting block of the FNO, they go into the model definition
problem.add_hyperparameter((1, 1024), "lifting_channels", default_value=256)
# 4 projection_channels:number of hidden channels of the projection block of the FNO, they go into the model definition
problem.add_hyperparameter((1, 1024), "projection_channels", default_value=64)
# 5. use_mlp: Whether to use an MLP layer after each FNO block, they go into the model definition
problem.add_hyperparameter([True, False], "use_mlp", default_value=False)
# 6 mlp['dropout']: parameter of the MLP, they go into the model definition
problem.add_hyperparameter((0.0, 1.0), "mlp['dropout']", default_value=0)
# 7 mlp['expansion']: MLP parameter, they go into the model definition
problem.add_hyperparameter((0.0, 3.0), "mlp['expansion']", default_value=0.5)
# 8 rank Rank of the tensor factorization of the Fourier weights, they go into the model definition
problem.add_hyperparameter((0.0, 1.0), "rank", default_value=1.0)
# 9 factorization:Tensor factorization of the parameters weight to use, they go into the model definition
problem.add_hyperparameter(
    ["tucker", "cp", "tt", "dense"], "factorization", default_value="dense"
)
# 10 learning_rate: this is the learning_rate, goes into the optimizer definition
problem.add_hyperparameter(
    (1e-6, 1e-2, "log-uniform"), "opt_learning_rate", default_value=5e-3
)
# 11 batch_size: the batch size for the training, goes into the train_loader for loading the datasel
problem.add_hyperparameter((2, 64), "data_batch_size", default_value=16)
# 12 weight_decay: this is the weight_decay, goes into the optimizer definition
problem.add_hyperparameter(
    (1e-6, 1e-2, "log-uniform"), "opt_weight_decay", default_value=1e-4
)
# 13 n_layers:Number of Fourier Layers
problem.add_hyperparameter((1, 8), "n_layers", default_value=4)
# 14 epochs: number of epochs, goes into the trainer
problem.add_hyperparameter((1, 1000), "opt_n_epochs", default_value=300)
# 15 scheduler_T_max: this goes into the scheduler, it's the max number of iterations in the scheduler
problem.add_hyperparameter([500], "opt_scheduler_T_max", default_value=500)
# 16 n_train: no of data samples to train the model, goes into the train_loader for loading the dataset
problem.add_hyperparameter([1000], "data_n_train", default_value=1000)
# the next set of hyperparameters are data related and we're keeping them constant for now,and for the particular problem
problem.add_hyperparameter([16], "data_train_resolution", default_value=16)
problem.add_hyperparameter([True], "data_positional_encoding", default_value=True)
problem.add_hyperparameter([True], "data_encode_input", default_value=True)
problem.add_hyperparameter([False], "data_encode_output", default_value=False)
problem.add_hyperparameter([True], "verbose", default_value=True)
problem.add_hyperparameter([666], "distributed_seed", default_value=666)
problem.add_hyperparameter([3], "data_channels", default_value=3)
# implementation is about how the factorization is done,
problem.add_hyperparameter(
    ["factorized", "reconstructed"], "implementation", default_value="factorized"
)
# By default, None, otherwise tanh is used before FFT in the FNO block
problem.add_hyperparameter(["None", "tanh"], "stabilizer", default_value=None)
# if 'full', the FNO Block runs in full precision,
# if 'half', the FFT, contraction, and inverse FFT run in half precision
# if 'mixed', the contraction and inverse FFT run in half precision
problem.add_hyperparameter(
    ["full", "half", "mixed"], "fno_block_precision", default_value="half"
)
# Type of skip connection to use,
problem.add_hyperparameter(
    ["soft-gating", "identity", "linear"], "skip", default_value="linear"
)
# type of training loss to use either H1Loss or Lploss
problem.add_hyperparameter(["h1", "l2"], "opt_training_loss", default_value="h1")
# the scheduler_patience, used for only  ReduceLROnPlateau, we're keeping it constant
problem.add_hyperparameter([5], "opt_scheduler_patience", default_value=5)
# type of scheduler to use
problem.add_hyperparameter(
    ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
    "opt_scheduler",
    default_value="StepLR",
)
# step_size for the optimizer
problem.add_hyperparameter((10, 100), "opt_step_size", default_value=60)
# gamma for the constant, we're keeping it constant
problem.add_hyperparameter([0.5], "opt_gamma", default_value=0.5)


@profile
def run(config):
    # important to avoid memory explosion
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, history = build_and_train_model(config, device, verbose=0)
    score = {
        "objective": -history["val_loss"][-1],
        "metadata": {
            "train_err": history["train_err"][-1],
            "num_parameters": history["num_parameters"],
            "budget": history["Epochs"][-1],
            "val_loss": history["val_loss"][-1],
            "duration_train": sum(history["train_time"]),
        },
    }
    score["objective"] = [score["objective"], -score["metadata"]["train_err"]]

    return score
