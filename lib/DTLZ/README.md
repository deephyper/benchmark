
# Modified Multiobjective DTLZ Test Suite

✅ **Benchmark ready to be used.**

This module contains objective function implementations of the DTLZ test
suite, derived from the implementations in
[ParMOO](https://github.com/parmoo/parmoo).

------------------------------------------------------------------------------

For further reference, the DTLZ test suite was originally proposed in:

    Deb, Thiele, Laumanns, and Zitzler. "Scalable test problems for
    evolutionary multiobjective optimization" in Evolutionary Multiobjective
    Optimization, Theoretical Advances and Applications, Ch. 6 (pp. 105--145).
    Springer-Verlag, London, UK, 2005. Abraham, Jain, and Goldberg (Eds).

The original implementation was appropriate for testing randomized algorithms,
but for many deterministic algorithms, the global solutions represent either
best- or worst-case scenarios, so an configurable offset was introduced in:

    Chang. "Mathematical Software for Multiobjective Optimization Problems."
    Ph.D. dissertation, Virginia Tech, Dept. of Computer Science, 2020.

Note that the DTLZ problems are minimization problems. Since DeepHyper
maximizes, the implementation herein returns the negative value for each of
the DTLZ objectives.

Our performance evaluator ``metrics`` scripts can evaluate either the
positive or negative solutions to estimate how well we have solved the
problem.

------------------------------------------------------------------------------

The full list of public classes in this module includes the 7 unconstrained
DTLZ problems
 * ``dtlz1``,
 * ``dtlz2``,
 * ``dtlz3``,
 * ``dtlz4``,
 * ``dtlz5``,
 * ``dtlz6``, and
 * ``dtlz7``

which are selected by setting the environment variable
``DEEPHYPER_BENCHMARK_DTLZ_PROB``.

## Installation 

To use the benchmark follow this example set of instructions:

```python

# Set DTLZ problem environment variables before loading
import os
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = "5" # 5 vars
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = "3" # 2 objs
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "2" # DTLZ2 problem
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.6" # soln [x_o, .., x_n]=0.6

# Load DTLZ benchmark suite
import deephyper_benchmark as dhb
dhb.load("DTLZ")

# Example of running one evaluation of DTLZ problem
from deephyper.evaluator import RunningJob
config = dtlz.hpo.problem.default_configuration # get a default config to test
res = dtlz.hpo.run(RunningJob(parameters=config))

```

## Configuration

To configure the problem, set the following:

- Environment variable `DEEPHYPER_BENCHMARK_PROB` with a value of `1`, `2`, ... `7` to select the DTLZ problem to run. Defaults to `2`.
- Environment variable `DEEPHYPER_BENCHMARK_NDIMS` with an integer value to set the number of input variables. Defaults to `5`.
- Environment variable `DEEPHYPER_BENCHMARK_NOBJS` with an integer value to set the number of objectives. Defaults to `2`
- Environment variable `DEEPHYPER_BENCHMARK_OFFSET` with a value between `0.0` and `1.0` to select the offset of the solution to the DTLZ problem. Defaults to `0.5` for DTLZ1, ..., DTLZ5 and `0.0` for DTLZ6 and DTLZ7. One may wish to adjust this value when comparing against deterministic blackbox solvers, which may sample the center and boundaries of the input space on specific schedules.
- Environment variable `DEEPHYPER_BENCHMARK_FAILURES` with value `0` or `1` to activate or deactivate failures. Defaults to `0`.

## Metadata

Since these problems are analytic, there is no metadata for this problem
beyond the standard DeepHyper metadata (timestamp information).

## Evaluating Results

Evaluating the performance of a multiobjective solver is nontrivial.
Typically, one should evaluate on two orthogonal bases:
 1. Quality of solutions -- What is the (average) error in the solutions
    returned by the solver?
 2. Diversity of solutions -- How much of the true Pareto front is covered
    by these solutions?

To evaluate these two metrics, we use:
 1. Improved generational distance (GD+): Let $F_i$ be a point in the solution
    set returned by a solver,
    and let $Y_i$ be the nearest point to $F_i$ on the true Pareto front,
    for $i=1,\ldots, n$.
    Then the GD+ is $\sum_{i} D^+(F_i, Y_i) / n$.
    Where $D^+$ denotes the improved distance function
    $D^+(F, Y) = ||\max(F_j - Y_j, 0)||_2^2$, where the "max" is taken
    componentwise.
    **Note that this metric may not increase monotonically. Additionally,
    it may be impossible to calculate for an arbitrary blackbox function,
    and can only be calculated here since the solution known and easily
    expressed algebraically for all of the DTLZ problems.**
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
we can calculate each of these metrics, and both the ``gdPlus(results)`` and
``hypervolume(results)`` functions are implemented in the ``dtlz.metrics``
module.
