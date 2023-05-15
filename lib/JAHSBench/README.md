
# JAHS Benchmark Suite

This module contains a DeepHyper wrapper for
 [JAHS-Bench-201](https://github.com/automl/jahs_bench_201).

------------------------------------------------------------------------------

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

## Usage

To use the benchmark follow this example set of instructions:

```python

# Load JAHS-bench-201
import deephyper_benchmark as dhb
dhb.load("JAHSBench")

# Example of running one evaluation of DTLZ problem
from deephyper.evaluator import RunningJob
config = dtlz.hpo.problem.default_configuration # get a default config to test
res = dtlz.hpo.run(RunningJob(parameters=config))

```

## Evaluating Results

Evaluating the performance of a multiobjective solver is nontrivial.
Typically, one should evaluate on two orthogonal bases:
 1. Quality of solutions -- What is the (average) error in the solutions
    returned by the solver?
 2. Diversity of solutions -- How much of the true Pareto front is covered
    by these solutions?

To evaluate these two metrics, we use:
 1. RMSE: Let $F_i$ be a point in the solution set returned by a solver,
    and let $Y_i$ be the nearest point to $F_i$ on the true Pareto front,
    for $i=1,\ldots, n$.
    Then the RMSE is $\sqrt{\sum_{i} (F_i - Y_i)^2 / n}$.
 2. Hypervolume dominated: Let $F_i$ be defined as above, and let $R$ be
    a pre-determined reference point such that all $F_i$ dominate $R$.
    Then the hypervolume is given by the volume of the union of all
    hyperboxes $B_i$ whose largest vertex is $F_i$ and smallest vertex
    is $R$. The value (and usefulness) of the hypervolume metric is extremely
    sensitive to the choice of $R$. Therefore, for this problem, we choose
    $R$ to be the Nadir point for the true Pareto front. **Note that in order
    to use the Nadir point as the reference point, we must throw out every
    solution returned by the solver that is worse than the Nadir point. For
    extremely difficult problems, this can result in zero hypervolume if no
    solutions better than the Nadir point were found. This is most common
    for DTLZ1, DTLZ3, and DTLZ7.**

For a general problem, the two metrics listed above could be very difficult
to compute and many researchers will use the hypervolume with an overly
pessimistic reference point as a proxy for both quality and diversity.
However, in general, the hypervolume tends to promote diversity over quality.
For the DTLZ problems, since the shape of the true Pareto front is known,
we can calculate each of these metrics, and both the ``rmse(results)`` and
``hypervolume(results)`` functions are implemented in the ``dtlz.metrics``
module.
