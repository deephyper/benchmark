import sys
from timeit import default_timer

import matplotlib.pyplot as plt
import neuralop.mpu.comm as comm
import numpy as np
import torch
import wandb
from neuralop import H1Loss, LpLoss
from neuralop.models import TFNO2d
from neuralop.training.patching import MultigridPatching2D
from neuralop.utils import count_params

from .data import load_darcy_flow_small


# this is the trainer class that trains the neural operators,
# it is from the neuralop package, but we had to slightly
# modify it in order to
# 1. calculate the validation loss instead of the testing loss
# 2. return the history of the training, including the time for each epoch, the training loss for each epoch
# and the validation loss for each epoch
class Trainer:
    def __init__(
        self,
        model,
        n_epochs,
        wandb_log=True,
        device=None,
        mg_patching_levels=0,
        mg_patching_padding=0,
        mg_patching_stitching=True,
        log_test_interval=1,
        log_output=False,
        use_distributed=False,
        verbose=True,
    ):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        mg_patching_levels : int, default is 0
            if 0, no multi-grid domain decomposition is used
            if > 0, indicates the number of levels to use
        mg_patching_padding : float, default is 0
            value between 0 and 1, indicates the fraction of size to use as padding on each side
            e.g. for an image of size 64, padding=0.25 will use 16 pixels of padding on each side
        mg_patching_stitching : bool, default is True
            if False, the patches are not stitched back together and the loss is instead computed per patch
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is True
        """
        self.n_epochs = n_epochs
        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.mg_patching_levels = mg_patching_levels
        self.mg_patching_stitching = mg_patching_stitching
        self.use_distributed = use_distributed
        self.device = device

        if mg_patching_levels > 0:
            self.mg_n_patches = 2**mg_patching_levels
            if verbose:
                print(f"Training on {self.mg_n_patches ** 2} multi-grid patches.")
                sys.stdout.flush()
        else:
            self.mg_n_patches = 1
            mg_patching_padding = 0
            if verbose:
                print(f"Training on regular inputs (no multi-grid patching).")
                sys.stdout.flush()

        self.mg_patching_padding = mg_patching_padding
        self.patcher = MultigridPatching2D(
            model,
            levels=mg_patching_levels,
            padding_fraction=mg_patching_padding,
            use_distributed=use_distributed,
            stitching=mg_patching_stitching,
        )

    def train(
        self,
        train_loader,
        validation_loader,
        output_encoder,
        model,
        optimizer,
        scheduler,
        regularizer,
        training_loss=None,
        eval_losses=None,
    ):
        """Trains the given model on the given datasets"""
        n_train = len(train_loader.dataset)
        history = {"Epochs": [], "train_err": [], "train_time": [], "val_loss": []}
        # if not isinstance(test_loaders, dict):
        #   test_loaders = dict(test=test_loaders)

        if self.verbose:
            print(f"Training on {n_train} samples")
            # print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
            #     f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        if output_encoder is not None:
            output_encoder.to(self.device)

        if self.use_distributed:
            is_logger = comm.get_world_rank() == 0
        else:
            is_logger = True

        for epoch in range(self.n_epochs):
            avg_loss = 0
            avg_lasso_loss = 0
            model.train()
            t1 = default_timer()
            train_err = 0.0

            for idx, sample in enumerate(train_loader):
                x, y = sample["x"], sample["y"]

                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f"Training on raw inputs of size {x.shape=}, {y.shape=}")

                x, y = self.patcher.patch(x, y)

                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f".. patched inputs of size {x.shape=}, {y.shape=}")

                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()

                out = model(x)
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f"Raw outputs of size {out.shape=}")

                out, y = self.patcher.unpatch(out, y)
                # Output encoding only works if output is stiched
                if output_encoder is not None and self.mg_patching_stitching:
                    out = output_encoder.decode(out)
                    y = output_encoder.decode(y)
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f".. Processed (unpatched) outputs of size {out.shape=}")

                loss = training_loss(out.float(), y)

                if regularizer:
                    loss += regularizer.loss

                loss.backward()

                optimizer.step()
                train_err += loss.item()

                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1
            del x, y

            train_err /= n_train
            avg_loss /= self.n_epochs
            history["Epochs"].append(epoch)
            history["train_err"].append(train_err)
            history["train_time"].append(epoch_train_time)
            # if epoch % self.log_test_interval == 0:

            msg = f"[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"

            values_to_log = dict(
                train_err=train_err, time=epoch_train_time, avg_loss=avg_loss
            )
            # for loader_name, loader in test_loaders.items():
            # if epoch == self.n_epochs - 1 and self.log_output:
            #     to_log_output = True
            # else:
            #     to_log_output = False

            loader = validation_loader
            loader_name = ""
            errors = self.evaluate(
                model, eval_losses, loader, output_encoder, log_prefix=loader_name
            )

            for loss_name, loss_value in errors.items():
                msg += f", {loss_name}={loss_value:.4f}"
                values_to_log[loss_name] = loss_value
            history["val_loss"].append(values_to_log[loss_name])
            if regularizer:
                avg_lasso_loss /= self.n_epochs
                msg += f", avg_lasso={avg_lasso_loss:.5f}"

            if self.verbose and is_logger:
                print(msg)
                sys.stdout.flush()

                # Wandb loging
            if self.wandb_log and is_logger:
                for pg in optimizer.param_groups:
                    lr = pg["lr"]
                    values_to_log["lr"] = lr
                wandb.log(values_to_log, step=epoch, commit=True)
        return history

    def evaluate(
        self, model, loss_dict, data_loader, output_encoder=None, log_prefix=""
    ):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        model : model to evaluate
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        output_encoder : used to decode outputs if not None
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        model.eval()

        if self.use_distributed:
            is_logger = comm.get_world_rank() == 0
        else:
            is_logger = True

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample["x"], sample["y"]

                n_samples += x.size(0)

                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                x = x.to(self.device)

                out = model(x)

                out, y = self.patcher.unpatch(out, y, evaluation=True)

                if output_encoder is not None:
                    out = output_encoder.decode(out)

                if (it == 0) and self.log_output and self.wandb_log and is_logger:
                    if out.ndim == 2:
                        img = out
                    else:
                        img = out.squeeze()[0]
                    wandb.log(
                        {
                            f"image_{log_prefix}": wandb.Image(
                                img.unsqueeze(-1).cpu().numpy()
                            )
                        },
                        commit=False,
                    )

                for loss_name, loss in loss_dict.items():
                    errors[f"{log_prefix}_{loss_name}"] += loss(out, y).item()

        del x, y, out

        for key in errors.keys():
            errors[key] /= n_samples

        return errors


# this function takes an input configuration and uses this to build a TFNO model in a format accepted by
# the TFNO benchmark and then trains the model
def build_and_train_model(config: dict, device="cpu", verbose: bool = 0):
    """
    this function takes an input configuration and uses this to build a TFNO model in
    a format accepted by the TFNO benchmark and then trains the model
    """
    default_config = {

        # General
        # For computing compression
        # 'n_params_baseline': None,

        # Distributed computing
        'distributed_seed': 666,

        # FNO related
        "lifting_channels": 256,
        'data_channels': 3,
        'n_modes_height': 16,
        'n_modes_width': 16,
        'hidden_channels': 32,
        'projection_channels': 64,
        'n_layers': 4,
        'skip': 'linear',
        'implementation': 'factorized',

        "use_mlp": False,
        "mlp": {"dropout": 0, "expansion": 0.5},

        'factorization': None,
        'rank': 1.0,

        'stabilizer': None,
        'fno_block_precision': 'half',

        # Optimizer

        'opt_n_epochs': 5,
        'opt_learning_rate': 5e-3,
        # 'opt_training_loss': 'h1',
        'opt_weight_decay': 1e-4,

        'opt_scheduler_T_max': 500,  # For cosine only, typically take n_epochs
        'opt_scheduler_patience': 5,  # For ReduceLROnPlateau only
        'opt_scheduler': 'StepLR',  # Or 'CosineAnnealingLR' OR 'ReduceLROnPlateau'
        'opt_step_size': 60,
        'opt_gamma': 0.5,

        # Dataset related

        'data_batch_size': 16,
        'data_n_train': 1000,
        'data_train_resolution': 16,
        'data_n_tests': [100, 50],
        'data_test_resolutions': [16, 32],
        'data_test_batch_sizes': [16, 16],
        'data_positional_encoding': True,

        'data_encode_input': True,
        'data_encode_output': False,

        # Patching
        'patching': {'levels': 0},
        'patching_levels': 0,
        'patching_padding': 0,
        'patching_stitching': False,
    }

    default_config.update(config)
    config_name = 'default'

   

    # Loading the Darcy flow dataset

    (train_loader, validation_loader, test_loaders, output_encoder) = load_darcy_flow_small(
        n_train=default_config['data_n_train'], batch_size=default_config['data_batch_size'],
        positional_encoding=default_config['data_positional_encoding'],
        test_resolutions=default_config['data_test_resolutions'], n_tests=default_config['data_n_tests'],
        test_batch_sizes=default_config['data_test_batch_sizes'],
        encode_input=default_config['data_encode_input'], encode_output=default_config['data_encode_output'],
        )

    model = TFNO2d(data_channels=default_config["data_channels"], n_modes_height=default_config['n_modes_height'],
                   n_modes_width=default_config["n_modes_width"],
                   n_layers=default_config["n_layers"], lifting_channels=default_config["lifting_channels"],
                   hidden_channels=default_config["hidden_channels"],
                   projection_channels=default_config["projection_channels"],
                   factorization=default_config["factorization"], rank=default_config["rank"],
                   use_mlp=default_config["use_mlp"], mlp=default_config["mlp"], skip=default_config["skip"],
                   implementation=default_config["implementation"],
                   stabilizer=default_config["stabilizer"], fno_block_precision=default_config["fno_block_precision"])
    model = model.to(device)

    # Log parameter count
    n_params = count_params(model)


    print(f'\nn_params: {n_params}')
    sys.stdout.flush()


    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=default_config['opt_learning_rate'],
                                 weight_decay=default_config['opt_weight_decay'])

    if default_config['opt_scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=default_config['opt_gamma'],
                                                               patience=default_config['opt_scheduler_patience'],
                                                               mode='min')
    elif default_config['opt_scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=default_config['opt_scheduler_T_max'])
    elif default_config['opt_scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=default_config['opt_step_size'],
                                                    gamma=default_config['opt_gamma'])
    else:
        raise ValueError(f"Got { default_config['opt_scheduler']= }")

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = dict(h1=train_loss)
    # if default_config['opt_training_loss'] == 'l2':
    #     train_loss = l2loss
    #     eval_losses = dict(l2=train_loss)
    # elif default_config['opt_training_loss'] == 'h1':
    #     train_loss = h1loss
    #     eval_losses = dict(h1=train_loss)
    # else:
    #     raise ValueError(f"Got training_loss={default_config['opt_training_loss']} but expected one of ['l2', 'h1']")
    # # eval_losses = {'h1': h1loss, 'l2': l2loss}


    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    print(f'\n### Beginning Training...\n')
    sys.stdout.flush()

    trainer = Trainer(model, n_epochs=default_config['opt_n_epochs'], device=device,
                      mg_patching_levels=default_config['patching_levels'],
                      mg_patching_padding=default_config['patching_padding'],
                      mg_patching_stitching=default_config['patching_stitching'], wandb_log=False, log_test_interval=1,
                      log_output=True, use_distributed=False, verbose=True)

    history = trainer.train(train_loader, validation_loader, output_encoder, model, optimizer, scheduler,
                            regularizer=False, training_loss=train_loss, eval_losses=eval_losses, )

    history["num_parameters"] = n_params

    return model, history



def bestevaluate(best_config):
    """we design the model based on the best hyperparameter configuration,
    we then evaluate this based on the training/validation and testing data"""

    (
        train_loader,
        validation_loader,
        test_loaders,
        output_encoder,
    ) = load_darcy_flow_small(
        n_train=best_config["data_n_train"],
        batch_size=best_config["data_batch_size"],
        positional_encoding=best_config["data_positional_encoding"],
        test_resolutions=[16, 32],
        n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        encode_input=best_config["data_encode_input"],
        encode_output=best_config["data_encode_output"],
    )
    # the build_and_train_model function
    # gives the training and validation losses
    best_model, best_history = build_and_train_model(best_config, verbose=1)
    # next we generate the testing losses
    trainer = Trainer(
        best_model,
        n_epochs=best_config["opt_n_epochs"],
        device=device,
        mg_patching_levels=best_config["patching_levels"],
        mg_patching_padding=best_config["patching_padding"],
        mg_patching_stitching=best_config["patching_stitching"],
        wandb_log=best_config["wandb_log"],
        log_test_interval=best_config["wandb_log_test_interval"],
        log_output=best_config["wandb_log_output"],
        use_distributed=best_config["distributed_use_distributed"],
        verbose=True,
    )
    for loader_name, loader in test_loaders.items():
        l2loss = LpLoss(d=2, p=2)
        h1loss = H1Loss(d=2)
        eval_losses = {"h1": h1loss, "l2": l2loss}
        errors = trainer.evaluate(
            best_model, eval_losses, loader, output_encoder, log_prefix=loader_name
        )
        for loss_name, loss_value in errors.items():
            best_history[loss_name] = loss_value

    test_samples = test_loaders[32].dataset

    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index]
        # Input x
        x = data["x"]
        # Ground-truth
        y = data["y"]
        # Model prediction
        out = best_model(x.unsqueeze(0))

        ax = fig.add_subplot(4, 4, index * 4 + 1)
        ax.imshow(x[0], cmap="gray")
        if index == 0:
            ax.set_title("Input x")
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(4, 4, index * 4 + 2)
        ax.imshow(y.squeeze())
        if index == 0:
            ax.set_title("Ground-truth y")
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(4, 4, index * 4 + 3)
        ax.imshow(out.squeeze().detach().numpy())
        if index == 0:
            ax.set_title("Model prediction")
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(4, 4, index * 4 + 4)
        err = (np.abs(y.squeeze() - out.detach().numpy())) ** 2
        im = ax.imshow(err.squeeze(), cmap="plasma", aspect="auto")
        plt.colorbar(im, label="$|y - \hat{y}|^2$")
        if index == 0:
            ax.set_title("Squared Error")
        plt.xticks([], [])
        plt.yticks([], [])

    fig.suptitle("Inputs, ground-truth output, prediction and absolute error.", y=0.98)
    plt.tight_layout()
    fig.show()
    return best_history
