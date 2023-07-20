import torch
from deepxde.nn import NN
from deepxde.nn import activations, initializers
from deepxde import config


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        skip_connection,
        dropout_rate,
        weight_decay,
    ):
        super().__init__()
        self.regularizer = ["l2", weight_decay]

        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            if skip_connection == "True":
                self.linears.append(SkipConnection(layer_sizes[i - 1], layer_sizes[i], initializer))
                
            else:
                self.linears.append(
                    torch.nn.Linear(
                        layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                    )
                )
                initializer(self.linears[-1].weight)
                initializer_zero(self.linears[-1].bias)
            self.linears.append(torch.nn.Dropout(p=dropout_rate))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class SkipConnection(torch.nn.Module):
    def __init__(self, in_dim, out_dim, initializer, hidden_dim=20) -> None:
        super().__init__()
        self.map = torch.nn.Linear(in_dim, out_dim)
        self.F = torch.nn.Linear(out_dim, hidden_dim)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden_dim, out_dim)
        self.bn2 = torch.nn.BatchNorm1d(out_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                initializer(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.in_dim != self.out_dim:
            x = self.map(x)
        residual = x
        x = self.F(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.bn2(x)
        out = residual + x
        return out
