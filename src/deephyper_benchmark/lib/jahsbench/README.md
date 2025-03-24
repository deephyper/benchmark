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
 - ``colorectal_histology``

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
  `fashion_mnist` (default), `cifar10`, or `colorectal_histology`. 

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
computing hypervolume scores for the Fashion MNIST dataset,
no accuracies less than 90% are considered for the CIFAR-10 dataset,
and no accuracies less than 93% are considered for the colorectal histology
dataset.
When solving any of these problems via DeepHyper, the ``moo_lower_bounds``
argument should be set accordingly.
To evaluate hypervolume with these reference points, use our metrics as
shown below

```python

from deephyper_benchmark.lib.jahsbench import metrics
evaluator = metrics.PerformanceEvaluator()
hv = evaluator.hypervolume(res)

```

## Additional Problem Details

Each problem in JAHS-Bench-201 is given by a XGBoost **surrogate** for a
joint neural network architecture/hyperparameter search problem.
Surrogates are given for networks trained to solve image classification tasks
for three different datasets.
These datasets are listed below and have the following properties:

- ``fashion_mnist`` is based on the Fashion MNIST dataset (https://github.com/zalandoresearch/fashion-mnist).
  This problem has lower bound of 95% on the "interesting range" of validation accuracies, and the Kendall's tau for the XGBoost model is over 92% for all 3 objectives (92.2% for the ``valid-acc`` objective).
- ``cifar10`` is based on the Cifar-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html)
  This problem has lower bound of 90% on the "interesting range" of validation accuracies, and the Kendall's tau for the XGBoost model is over 89% for all 3 objectives (89% for the ``valid-acc`` objective).
- ``colorecal-histology`` is based on the colorectal histology dataset (https://zenodo.org/record/53169#.XGZemKwzbmG).
  This problem has lower bound of 93% on the "interesting range" of validation accuracies, and the Kendall's tau for the XGBoost model is considerably lower than with the other datasets (just 68.2% for the ``valid-acc`` objective), so it may not be an accurate representation of the true problem.
