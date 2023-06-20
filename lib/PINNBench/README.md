# Physics-informed Neural Networks Benchmark
Physics-Informed Neural Networks (PINNs) are a class of machine learning models that combine the strengths of neural networks and physics-based modeling. PINNs are used to solve partial differential equations (PDEs) and other physical problems by learning a solution directly from data.

The basic idea behind PINNs is to use a neural network to approximate the solution to a PDE, while also enforcing the underlying physical laws that govern the problem. This is achieved by incorporating the PDE as a constraint in the neural network training process. More details can be found in the [original work](https://arxiv.org/abs/1711.10561).

This set of benchmarks seek to incorporate AutoML workflow into the development of PINNs with DeepHyper. The PINN benchmark problems support Hyperparameter Optimization (HPO), Neural Architecture Search (NAS), and Multi-fidelity evaluations. 

The current available problems are 

[Burgers Equation](#burgers-equation)

[Diffusion-reaction Equation](#diffusion-reaction-equation)


## Burgers Equation
### Installation
To install

```
python -c "import deephyper_benchmark as dhb; dhb.install('PINNBench/Burgers');"
```

<!-- PINNs are neural networks are a type of machine learning method that combines deep neural networks with physical equations to solve complex physical problems. In the case of 1-D Burgers Equation, training a PINN, $\hat{u}$, is to minimize a compound loss function. The 1-D Burgers equation has the following form:

$$\frac{\partial u }{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

The loss functions has two component,
$$L_{tot} = L_u + \alpha L_f$$

the solution loss for specified initial and boundary conditions,
$$L_u = \vert \vert \hat{u} - u \vert \vert^2_2$$

and the loss according to the governing equation (PDE loss) for every other points in the domain,
$$L_f  = \vert \vert \frac{\partial \hat{u}}{\partial t} + \hat{u}\frac{\partial \hat{u}}{\partial x} - \nu \frac{\partial^2 \hat{u}}{\partial x^2})\vert \vert^2_2.$$ -->


### Hyperparameter Search
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
### Defualt configuration and results
The default configuration of the PINN.

| | |
|:--|:--|
|`num_layers`| 5 |
|`lr` | 0.01|
| `hidden_dim`| 5|
|`alpha`| 5 |
|`activation`| `tanh`|

```
result={'objective': -0.06480624, 'metadata': {'timestamp_start': 1680036315.47377, 'timestamp_end': 1680036380.2708638, 'num_parameters': 171, 'train_loss': 0.06834503, 'val_loss': 0.06480624, 'test_loss': 0.066540696, 'budget': 1000, 'stopped': False, 'infos_stopped': None}}
```



## Diffusion-reaction Equation
This benchmark is based on **modified** [`PDEBench`](https://github.com/pdebench/PDEBench) and [`DeepXDE`](https://github.com/lululxvi/deepxde). 

### Installation

To install the **modified** `PDEBench` and this benchmark
```
python -c "import deephyper_benchmark as dhb; dhb.install('PINNBench/Diffusion-reaction');"
```

Run the hyperparameter search
```
import os
os.environ['DEEPHYPER_BENCHMARK_DATASET'] = '2D_diff-react_NA_NA' # set benchmark dataset

import deephyper_benchmark as dhb
diff_react = dhb.load("PINNBench/Diffusion-reaction")

from deephyper.evaluator import RunningJob
config = diff_react.hpo.problem.default_configuration # get a default config to test
res = diff_react.hpo.run(RunningJob(parameters=config))
```

### Configuration
It is necessary to configure `DeepXDE` to use `PyTorch` backend. The instructions can be found [here](https://deepxde.readthedocs.io/en/latest/user/installation.html#working-with-different-backends).
