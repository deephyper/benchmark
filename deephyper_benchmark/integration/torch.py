import numpy as np
import torch.nn as nn


def count_params(module: nn.Module) -> dict:
    """Evaluate the number of parameters of a Torch module.

    Args:
        module(nn.Module): a Torch module.

    Returns:
        dict: a dictionary with the number of trainable ``"num_parameters_train"`` and
        non-trainable parameters ``"num_parameters"``.
    """
    num_parameters = int(np.sum(p.numel() for p in module.parameters()))
    num_parameters_train = int(
        np.sum(p.numel() for p in module.parameters() if p.requires_grad)
    )
    return {
        "num_parameters": num_parameters,
        "num_parameters_train": num_parameters_train,
    }
