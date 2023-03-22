# DeepHyper Benchmark

## Table of Contents

* [Table of Contents](#table-of-contents)
* [Introduction](#introduction)
* [Organization of the Repository](#organization-of-the-repository)

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

## List of Benchmarks

| Name       | Description                                                                  | Variable(s) Type                             | Objective(s) Type | Multi-Objective | Multi-Fidelity | Evaluation Duration |
| ---------- | ---------------------------------------------------------------------------- | -------------------------------------------- | ----------------- | --------------- | -------------- | ------------------- |
| C-BBO      | Continuous Black-Box Optimization problems.                                  | $\mathbb{R}$                                 | $\mathbb{R}$      | ❌              | ❌             | ms                  |
| ECP-Candle | Deep Neural-Networks on multiple "biological" scales of Cancer related data. | $\mathbb{R}\times\mathbb{N}\times\mathbb{C}$ | $\mathbb{R}$      | ❌              | ❌             | min                 |
| HPOBench   | Hyperparameter Optimization Benchmark.                                       | $\mathbb{R}\times\mathbb{N}\times\mathbb{C}$ | $\mathbb{R}$      | ❌              | ✅             | ms to min           |
| LCu        | Learning curve hyperparameter optimization benchmark.                        |                                              |                   |                 |                |                     |
| PINNBench  | Physics Informed Neural Networks Benchmark.                                  |                                              |                   |                 |                |                     |
| Toy        | Toy examples for debugging.                                                  |                                              |                   |                 |                |                     |
|            |                                                                              |                                              |                   |                 |                |                     |