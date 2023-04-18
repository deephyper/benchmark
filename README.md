# DeepHyper Benchmark

## Table of Contents

- [DeepHyper Benchmark](#deephyper-benchmark)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Organization of the Repository](#organization-of-the-repository)
  - [Installation](#installation)
  - [Defining a Benchmark](#defining-a-benchmark)
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

After cloning the repository:
```python
pip install -e.
```

## Defining a Benchmark

A benchmark is defined as a sub-folder of the `lib/` folder such as `lib/Benchmark-101/`. Then a benchmark folder needs to follow a python package structure and therefore it needs to contain a `__init__.py` file at its root. In addition, a benchmark folder needs to define a `benchmark.py` script which defines its requirements.

General benchmark structure:
```
lib/
    Benchmark-101/
        __init__.py
        benchmark.py
        data.py
        model.py
        hpo.py # defines hyperparameter optimization inputs (run-function + problem)
```

Then to use the benchmark:

```python
from deephyper_benchmark import *

install("Benchmark-101")

load("Benchmark-101")

from deephyper_benchmark.lib.benchmark_101.hpo import problem, run
```

All `run`-functions (i.e., function returning the objective(s) to be optimized) should follow the **MAXIMIZATION** standard. If a benchmark needs minimization then the negative of the minimized objective can be returned `return -minimized_objective`.

## Standard Metadata

Benchmarks must return the following standard metadata when it applies, some metadata are specific to neural networks (e.g., `num_parameters`):

- `num_parameters`: integer value of the number of trainable parameters of the neural network.
- `budget`: scalar value (float/int) of the budget consumed by the neural network. Therefore the budget should be defined for each benchmark (e.g., number of epochs in general).
- `stopped`: boolean value indicating if the evaluation was stopped before consuming the maximum budget.
- `train_X`:  scalar value of the training metrics (replace `X` by the metric name, 1 key per metric).
- `valid_X`: scalar value of the validation metrics (replace `X` by the metric name, 1 key per metric).
- `test_X`: scalar value of the testing metrics (replace `X` by the metric name, 1 key per metric).


The `@profile` decorator should be used on all `run`-functions to collect the `timestamp_start` and `timestamp_end` metadata.

## List of Benchmarks

| Name       | Description                                                                  | Variable(s) Type                             | Objective(s) Type | Multi-Objective | Multi-Fidelity | Evaluation Duration |
| ---------- | ---------------------------------------------------------------------------- | -------------------------------------------- | ----------------- | --------------- | -------------- | ------------------- |
| C-BBO      | Continuous Black-Box Optimization problems.                                  | $\mathbb{R}$                                 | $\mathbb{R}$      | ❌              | ❌             | ms                  |
| ECP-Candle | Deep Neural-Networks on multiple "biological" scales of Cancer related data. | $\mathbb{R}\times\mathbb{N}\times\mathbb{C}$ | $\mathbb{R}$      | ❌              | ❌             | min                 |
| HPOBench   | Hyperparameter Optimization Benchmark.                                       | $\mathbb{R}\times\mathbb{N}\times\mathbb{C}$ | $\mathbb{R}$      | ❌              | ✅             | ms to min           |
| LCu        | Learning curve hyperparameter optimization benchmark.                        |                                              |                   |                 |                |                     |
| LCbench    | Multi-fidelity benchmark without hyperparameter optimization.                | NA                                           | $\mathbb{R}$      | ❌              | ✅             | secondes            |
| PINNBench  | Physics Informed Neural Networks Benchmark.                                  | $\mathbb{R}^2$                               | $\mathbb{R}$      | ❌              | ✅             | ms                  |
| Toy        | Toy examples for debugging.                                                  |                                              |                   |                 |                |                     |
| DTLZ       | The modified DTLZ multiobjective test suite.                                 |  $\mathbb{R}$                                |  $\mathbb{R}$     | ✅              |  ❌            | configurable        |
|                |                                                                          |                                              |                   |                 |                |                     |
      
      
      
      
      
      
      
  
      
      
      
      
      
      
  
      
      
      
      
