# JAHS Benchmark Suite

> **Warning**
> Work in progress, this benchmark is not yet ready.

This module contains a DeepHyper wrapper for
 [JAHS-Bench-201](https://github.com/automl/jahs_bench_201).

JAHSBench implements a random forest surrogate model, trained on real-world
performance data for neural networks trained on three standard benchmark
problems:
 - ``fashion_mnist`` (**default**)
 - ``cifar10`` 
 - ``colorectal_history``

Using these models as surrogates for the true performance, we can use this
benchmark problem to study the performance of AutoML techniques on joint
architecture-hyperparameter search tasks at minimal expense.

The models allow us to tune 2 continuous training hyperparameters
 - ``LearningRate`` and
 - ``WeightDecay``,

2 categorical training hyperparameters
 - ``Activation`` and
 - ``TrivialAugment``,

and 5 categorical architecture parameters
 - ``Op{i}`` for ``i=0, ..., 4``.

For DeepHyper's implementation, we have added 9th integer-valued parameter,
which is the number of epochs trained
 - ``nepochs``.

When run with the option ``wait=True``, ``JAHSBench`` will wait for an
amount of time proportional to the ``runtime`` field returned by
JAHS-Bench-201's surrogates. By default, this is 1% of the true runtime.

The benchmark can be run to tune a single objective (``valid-acc``) or
three objectives (``valid-acc``, ``latency``, and ``size_MB``).
*Note that in the original JAHS-Bench-201 benchmark, there are only 2
objectives and ``size_MB`` is not included as an objective.*

For further information, see:

```
    @inproceedings{NEURIPS2022_fd78f2f6,
        author = {Bansal, Archit and Stoll, Danny and Janowski, Maciej and Zela, Arber and Hutter, Frank},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
        pages = {38788--38802},
        publisher = {Curran Associates, Inc.},
        title = {JAHS-Bench-201: A Foundation For Research On Joint Architecture And Hyperparameter Search},
        url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/fd78f2f65881c1c7ce47e26b040cf48f-Paper-Datasets_and_Benchmarks.pdf},
        volume = {35},
        year = {2022}
    }
```

## Installation

To install this benchmark the following command can be run:
```
python -c "import deephyper_benchmark as dhb; dhb.install('JAHSBench');"
```

## Configuration

Prior to initialize the problem, set the following environment variables
to configure the JAHS-Bench problem:
- ``DEEPHYPER_BENCHMARK_MOO`` to `0` for single objective runs or `1` for
  multiobjective runs. Defaults to `1`.
- ``DEEPHYPER_BENCHMARK_JAHS_PROB`` to one of the following:
  `fashion_mnist` (default), `cifar10`, or `colorectal_history`. 

## Metadata

In addition to DeepHyper's standard metadata (timestamps), the following metadata
is produced by JAHS-Bench-201, and recorded by DeepHyper:
- ``m:size_MB``,
- ``m:runtime``,
- ``m:latency``,
- ``m:FLOPS``,
- ``m:valid-acc``,
- ``m:train-acc``, and
- ``m:test-acc``.

The ``valid-acc`` is used to compute the objective in the single objective
case.
Additionally, the negative values of the ``latency`` and ``size_MB`` are
used in the multiobjective case.
All metadata is recorded regardless of the case.

## Usage

To use the benchmark follow this example set of instructions:

```python

import deephyper_benchmark as dhb


# Load JAHS-bench-201
dhb.load("JAHSBench")

from deephyper_benchmark.lib.jahsbench import hpo

# Example of running one evaluation of JAHSBench
from deephyper.evaluator import RunningJob
config = hpo.problem.jahs_obj.__sample__() # get a default config to test
res = hpo.run(RunningJob(parameters=config))

```

Note that JAHS-Bench-201 uses XGBoost, which may not be compatible with older
versions of MacOS.
Additionally, the surrogate data has been pickled with an older version
of scikit-learn and newer versions will fail to correctly load the surrogate
models.

For more information, see the following GitHub issues:
 - https://github.com/automl/jahs_bench_201/issues/6
 - https://github.com/automl/jahs_bench_201/issues/18

## Evaluating Results

To evaluate the results, the AutoML team recommends using the validation
error for single-objective runs or the hypervolume metric over both
validation error and evaluation latency for multiobjective-runs.
See their
[Evaluation Protocol](https://automl.github.io/jahs_bench_201/evaluation_protocol)
for more details.

**For multiobjective problems:**
In their original benchmark, no recommended reference point is given,
as discussed in
[this GitHub issue](https://github.com/automl/jahs_bench_201/issues/19).
Since we have already modified the original problem, we have taken the
liberty of assigning a reference point.
This reference point was chosen to include all true solutions for the
``latency`` and ``size_MB`` objectives.
These values were chosen by observing the true Pareto front for the raw data.
However, we have purposefully created a lower bound on the "interesting range"
for the ``valid-acc`` objective, which is higher than it was before.
In particular, no accuracies less than 95% will be considered when
computing hypervolume scores for the Fashion MNIST or CIFAR-10 datasets,
and no accuracies less than 98% are considered for the colorectal screening
dataset.
To evaluate hypervolume with these reference points, use our metrics as
shown below

```python

from deephyper_benchmark.lib.jahsbench import metrics
evaluator = metrics.PerformanceEvaluator()
hv = evaluator.hypervolume(res)

```
