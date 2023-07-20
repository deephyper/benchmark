import torch
import torch.nn as nn

from deepxde.nn import NN
from deepxde.nn import activations, initializers
from deepxde import config


#LAAF : true false
# "LAAF-10 relu"

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
        skip_co: bool = False,
        dropout_rate: float = 0.0,
        regularization: str = None,
        weight_decay: float = 0.0,
        laaf: bool = False,
        laaf_scaling_factor: float = 10,

        **kwargs,
    ):
        super().__init__()
        self.regularizer = [regularization, weight_decay]

        layer_sizes = [input_dim] + [num_neurons for _ in range(num_layers)]

        if laaf:
            activation = f"LAAF-{laaf_scaling_factor} {activation}"
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = nn.Sequential()
        for i in range(1, len(layer_sizes)):
            if skip_co:
                self.linears.append(
                    SkipConnection(
                        in_dim=layer_sizes[i - 1],
                        out_dim=layer_sizes[i],
                        initializer=initializer,
                        activation=self.activation,
                    )
                )

            else:
                self.linears.append(
                    nn.Linear(
                        layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                    )
                )

                initializer(self.linears[-1].weight)
                initializer_zero(self.linears[-1].bias)

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
    def __init__(self, in_dim, out_dim, initializer, activation=nn.ReLU) -> None:
        super().__init__()
        self.block = nn.Sequential()

        self.F = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initializer(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        residual = x
        x = self.F(x)
        x = self.bn(x)
        x = self.activation(x)
        out = residual + x
        return out
