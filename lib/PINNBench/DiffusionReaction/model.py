import torch
import torch.nn as nn

from deepxde.nn import NN
from deepxde.nn import activations, initializers
from deepxde import config

ACTIVATIONS = activations
INITIALIZERS = initializers

# LAAF : true false
# "LAAF-10 relu"


class Activation(nn.Module):
    def __init__(self, func) -> None:
        super(Activation, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


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
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()

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
                self.linears.append(Activation(func=ACTIVATIONS.get(activation)))

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
        self.block = nn.Sequential()
        self.initializer = INITIALIZERS.get(kernel_initializer)

        if in_dim != out_dim:
            self.map = nn.Linear(in_dim, out_dim)

        linear_module = nn.Linear(in_dim, out_dim)
        self.initializer(linear_module.weight)
        self.initializer_zero(linear_module.bias)
        self.block.append(linear_module)
        if batch_norm:
            self.block.append(nn.BatchNorm1d(out_dim))
        self.block.append(Activation(func=activation))

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        residual = x
        if self.in_dim != self.out_dim:
            residual = self.map(residual)
        out = self.block(x) + residual
        return out
