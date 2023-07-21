import torch
import torch.nn as nn

from deepxde.nn import NN
from deepxde.nn import initializers
from deepxde import config

INITIALIZERS = initializers

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

ACTIVATIONS = {
        # "id": nn.Identity,
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU, # same as swish
        "sin": Sin, 
        "tanh": nn.Tanh,
        "hardswish": nn.Hardswish,
        "leakyrelu": nn.LeakyReLU,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
}


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 5,
        num_neurons: int = 30,
        activation: str = "elu",
        kernel_initializer: str = "Glorot normal",
        batch_norm: bool = False,
        skip_co: bool = False,
        dropout_rate: float = 0.0,
        regularization: str = None,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super(FNN, self).__init__()

        if regularization is None:
            self.regularizer = None
        else:
            self.regularizer = [regularization, weight_decay]

        layer_sizes = [input_dim] + [num_neurons for _ in range(num_layers)]

        initializer = INITIALIZERS.get(kernel_initializer)
        initializer_zero = INITIALIZERS.get("zeros")

        self.linears = nn.Sequential()
        for i in range(1, len(layer_sizes)):
            if skip_co:
                self.linears.append(
                    SkipConnection(
                        in_dim=layer_sizes[i - 1],
                        out_dim=layer_sizes[i],
                        batch_norm=batch_norm,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                    )
                )

            else:
                linear_module = nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
                initializer(linear_module.weight)
                initializer_zero(linear_module.bias)
                self.linears.append(linear_module)
                if batch_norm:
                    self.linears.append(
                        nn.BatchNorm1d(layer_sizes[i], dtype=config.real(torch))
                    )
                
                if activation != "id":
                    self.linears.append(ACTIVATIONS.get(activation)())

            self.linears.append(nn.Dropout(p=dropout_rate))

        self.linears.append(
            nn.Linear(layer_sizes[-1], output_dim, dtype=config.real(torch))
        )

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        x = self.linears(x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class SkipConnection(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_initializer="Glorot normal",
        batch_norm=False,
        activation="elu",
    ) -> None:
        super(SkipConnection, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        initializer = INITIALIZERS.get(kernel_initializer)
        initializer_zero = INITIALIZERS.get("zeros")

        if self.in_dim != self.out_dim:
            self.map = nn.Linear(in_dim, out_dim)
            initializer(self.map.weight)
            initializer_zero(self.map.bias)

        self.block = nn.Sequential()
        
        linear_module = nn.Linear(in_dim, out_dim, dtype=config.real(torch))
        initializer(linear_module.weight)
        initializer_zero(linear_module.bias)
        self.block.append(linear_module)

        if batch_norm:
            self.block.append(nn.BatchNorm1d(out_dim, dtype=config.real(torch)))

        if activation != "id":
            self.block.append(ACTIVATIONS.get(activation)())
        
        if activation != "id":
            self.act = ACTIVATIONS.get(activation)()
        else:
            self.act = None

    def forward(self, x):
        residual = x
        if self.in_dim != self.out_dim:
            residual = self.map(residual)
        out = self.block(x) + residual
        if self.act is not None:
            out = self.act(out)
        return out
