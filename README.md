# DeepHyper Benchmark

## Table of Contents

* [Table of Contents](#table-of-contents)
* [Introduction](#introduction)
* [Organization of the Repository](#organization-of-the-repository)
* [Model of Execution](#model-of-execution)

## Introduction

This repository is a benchmark framework for DeepHyper.

## Organization of the Repository

The repository follows this organization:

```bash
# Python package containing utility code
src/deephyper_benchmark/

# Scripts to execute benchmarks
src/
    scripts/
        argonne-lcf/ # argonne lcf execution (theta-knl, theta-gpu, ...)
        local/ # local execution (laptopt, single instance)

# Test the good behaviour of the deephyper_benchmark package
tests/
```

## Installation

```python
pip install -e.
```

## Model of Execution

The entry-point is the `deephyper-benchmark` command line to which we provide a benchmark script to execute. This script is a Python file containing an implemented sub-class of `Benchmark`:

```console
deephyper-benchmark src/scripts/local/hps_101_ray_evaluator.py -v -o output/
```

Each Benchmark will be executed in 3 steps:

```python
benchmark.initialize()
benchmark.execute()
benchmark.report()
```
