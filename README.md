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

The entry-point is the `deephyper-benchmark` command line to which we provide a configuration to execute. This configuration is passed as a `.yaml` file containing all the information necessary to perform the run :

```yaml
script: benchmark/script/path/from/src/scripts/
database: json/file/path/relative/to/the/command/line

summary:
  user: user_name
  group: name of the experiment this run falls in
  label: name of the run in this group
  description: a quick description

env:
    system:
      name: system_name
      type: laptop | server | theta

parameters:
  param_1: value
  param_2: value
  param_3: value
```

The `script` and `database` fields refer to the paths, respectively, of the benchmark script which must be executed and the path to the database, which must be a `.json` file. The script path is relative to the `src/scripts/` directory, refer to the [repository organization](#organization-of-the-repository). This is not the case for the database path.

Once this configuration is correctly established, the following command can be executed to perform the run :

```console
deephyper-benchmark config.yaml -v
```

Each Benchmark will be executed in 3 steps:

```python
benchmark.initialize()
benchmark.execute()
benchmark.report()
```

These three functions need to be implemented when writing a new benchmark, also `benchmark.report()`has to return a dictionary containing the different results of the run.

In order to take parameters as inputs, a default configuration of these must be given in the script. The `benchmark.load_parameters()`function will then rewrite these default parameters using those that are given by the user ; this function can be overridden to perform castings and verifications over these entries.
