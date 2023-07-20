# DeepHyper Benchmark

## Table of Contents

- [DeepHyper Benchmark](#deephyper-benchmark)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Organization of the Repository](#organization-of-the-repository)
  - [Installation](#installation)
  - [Defining a Benchmark](#defining-a-benchmark)
  - [Standard Metadata](#standard-metadata)
  - [List of Benchmarks](#list-of-benchmarks)

## Introduction

This repository is a collection of machine learning benchmark for DeepHyper.

## Organization of the Repository

The repository follows this organization:

```bash
# Python package containing utility code
deephyper_benchmark/

# Library of benchmarks
lib/
```

## Installation

To install the DeepHyper benchmark suite, run:

```console
git clone https://github.com/deephyper/benchmark.git deephyper_benchmark
cd deephyper_benchmark/
pip install -e "."
```

## Defining a Benchmark

A benchmark is defined as a sub-folder of the `lib/` folder such as `lib/Benchmark-101/`. Then a benchmark folder needs to follow a python package structure and therefore it needs to contain a `__init__.py` file at its root. In addition, a benchmark folder needs to define a `benchmark.py` script that defines its requirements.

General benchmark structure:
```
lib/
    Benchmark-101/
        __init__.py
        benchmark.py
        data.py
        model.py
        hpo.py # Defines hyperparameter optimization inputs (run-function + problem)
        README.md # Description of the benchmark
```

Then to use the benchmark:

```python
import deephyper_benchmark as dhb

dhb.install("Benchmark-101")

dhb.load("Benchmark-101")

from deephyper_benchmark.lib.benchmark_101.hpo import problem, run
```

All `run`-functions (i.e., functions returning the objective(s) to be optimized) should follow the **MAXIMIZATION** standard. If a benchmark needs minimization then the negative of the minimized objective can be returned `return -minimized_objective`.

A benchmark inherits from the `Benchmark` class:

```python
import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class Benchmark101(Benchmark):

    version = "0.0.1"

    requires = {
        "bash-install": {"type": "cmd", "cmd": "cd .. && " + os.path.join(DIR, "../install.sh")},
    }

```

Finally, when testing a benchmark it can be useful to activate the logging:

```python
import logging

logging.basicConfig(
    # filename="deephyper.log", # Uncomment if you want to create a file with the logs
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)
```

## Configuration

Benchmarks can sometimes be configured. The configuration can use environment variables with the prefix `DEEPHYPER_BENCHMARK_`.

## Standard Metadata

Benchmarks must return the following standard metadata when it applies, some metadata are specific to neural networks (e.g., `num_parameters`):

- [ ] `num_parameters`: integer value of the number of parameters in the neural network.
- [ ] `num_parameters_train`: integer value of the number of **trainable** parameters of the neural network.
- [ ] `budget`: scalar value (float/int) of the budget consumed by the neural network. Therefore the budget should be defined for each benchmark (e.g., number of epochs in general).
- [ ] `stopped`: boolean value indicating if the evaluation was stopped before consuming the maximum budget.
- `train_X`:  scalar value of the training metrics (replace `X` by the metric name, 1 key per metric).
- `valid_X`: scalar value of the validation metrics (replace `X` by the metric name, 1 key per metric).
- `test_X`: scalar value of the testing metrics (replace `X` by the metric name, 1 key per metric).
- [ ] `flops`: number of flops of the model such as computed in `fvcore.nn.FlopCountAnalysis(...).total()` (See [documentation](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#module-fvcore.nn)).
- [ ] `latency`: TO BE CLARIFIED
- [ ] `lc_train_X`: recorded learning curves of the trained model, the `bi` variables are the budget value (e.g., epochs/batches), and the `yi` values are the recorded metric. `X` in `train_X` is replaced by the name of the metric such as `train_loss` or `train_accuracy`. The format is `[[b0, y0], [b1, y1], ...]`.
- [ ] `lc_valid_X`: Same as `lc_train_X` but for validation data.


The `@profile` decorator should be used on all `run`-functions to collect the `timestamp_start` and `timestamp_end` metadata.

## List of Benchmarks

In the following table:

- $\mathbb{R}$ denotes real parameters.
- $\mathbb{D}$ denotes discrete parameters.
- $\mathbb{C}$ denotes categorical parameters.

| Name       | Description                                                                  | Variable(s) Type                             | Objective(s) Type | Multi-Objective | Multi-Fidelity | Evaluation Duration |
| ---------- | ---------------------------------------------------------------------------- | -------------------------------------------- | ----------------- | --------------- | -------------- | ------------------- |
| C-BBO      | Continuous Black-Box Optimization problems.                                  | $\mathbb{R}^n$                               | $\mathbb{R}$      | ❌              | ❌             | configurable        |
| ECP-Candle | Deep Neural-Networks on multiple "biological" scales of Cancer related data. | $\mathbb{R}\times\mathbb{D}\times\mathbb{C}$ | $\mathbb{R}$      | ✅              | ✅             | min                 |
| HPOBench   | Hyperparameter Optimization Benchmark.                                       | $\mathbb{R}\times\mathbb{D}\times\mathbb{C}$ | $\mathbb{R}$      | ✅              | ✅             | ms to min           |
| LCu        | Learning curve hyperparameter optimization benchmark.                        |                                              |                   |                 |                |                     |
| LCbench    | Multi-fidelity benchmark without hyperparameter optimization.                | NA                                           | $\mathbb{R}$      | ❌              | ✅             | secondes            |
| PINNBench  | Physics Informed Neural Networks Benchmark.                                  | $\mathbb{R}\times\mathbb{D}\times\mathbb{C}$ | $\mathbb{R}$      | ✅              | ✅             | ms                  |
| Toy        | Toy examples for debugging.                                                  |                                              |                   |                 |                |                     |
| DTLZ       | The modified DTLZ multiobjective test suite.                                 |  $\mathbb{R}^n$                              |  $\mathbb{R}$     | ✅              |  ❌            | configurable        |
| JAHSBench  | A slightly modified JAHSBench 201 wrapper.                                   |  $\mathbb{R}^2\times\mathbb{D}\times\mathbb{C}^8$ | $\mathbb{R}$ | ✅              |  ❌            | configurable        |
|            |                                                                              |                                              |                   |                 |                |                     |
      
      
      
      
      
      
      
  
      
      
      
      
      
      
  
      
      
      
      
