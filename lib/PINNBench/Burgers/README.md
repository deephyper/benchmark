
# PINN Benchmark - Burgers Equation

This benchmark is based on the implementation of Physics-informed Neural networks (PINN) (https://arxiv.org/abs/1711.10561) on 1D Burgers Equation. 

To use the benchmark follow this example set of instructions:

```python
from deephyper_benchmark import *

install("PINNBench/Burgers")

load("PINNBench/Burgers")

from deephyper_benchmark.lib.PINNBench.Burgers.hpo import run
```

PINNs are neural networks are a type of machine learning method that combines deep neural networks with physical equations to solve complex physical problems. In the case of 1-D Burgers Equation, training a PINN, $\hat{u}$, is to minimize a compound loss function. The 1-D Burgers equation has the following form:

$$\frac{\partial u }{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

The loss functions has two component,
$$L_{tot} = L_u + \alpha L_f$$

the solution loss for specified initial and boundary conditions,
$$L_u = \vert \vert \hat{u} - u \vert \vert^2_2$$

and the loss according to the governing equation (PDE loss) for every other points in the domain,
$$L_f  = \vert \vert \frac{\partial \hat{u}}{\partial t} + \hat{u}\frac{\partial \hat{u}}{\partial x} - \nu \frac{\partial^2 \hat{u}}{\partial x^2})\vert \vert^2_2.$$


## Hyperparameter Search
The objective of this benchmark is to optimize a set of hyperparameters for a feedforward neural network, including `num_layers`, `lr` (the learning rate of the optimizer), `hidden_dim` (the number of neurons in the hidden layers), `alpha` (the PDE loss coefficient), and `activation` (the activation function). The optimization is performed by minimizing the negative total loss value from the validation dataset.

To improve the efficiency of the search, the benchmark uses MCModelStopper to predict the training dynamics and perform early stopping when the predicted subsequent iterations are unlikely to lead to a further improvement in model performance. The maximum budget, `max_b`, for this search is 1000 epochs.

```python
for budget_i in range(min_b, max_b + 1):
        train_loss, eval = sup.step(train, val)
        objective_i = -eval  # maximizing in deephyper
        job.record(budget_i, objective_i)
        if job.stopped():
            break
```