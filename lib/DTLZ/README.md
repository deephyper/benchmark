
# Modified Multiobjective DTLZ Test Suite

This module contains objective function implementations of the DTLZ test
suite, derived from the implementations in
[ParMOO](https://github.com/parmoo/parmoo).

------------------------------------------------------------------------------

For further references, the DTLZ test suite was originally proposed in:

    Deb, Thiele, Laumanns, and Zitzler. "Scalable test problems for
    evolutionary multiobjective optimization" in Evolutionary Multiobjective
    Optimization, Theoretical Advances and Applications, Ch. 6 (pp. 105--145).
    Springer-Verlag, London, UK, 2005. Abraham, Jain, and Goldberg (Eds).

The original implementation was appropriate for testing randomized algorithms,
but for many deterministic algorithms, the global solutions represent either
best- or worst-case scenarios, so an configurable offset was introduced in:

    Chang. "Mathematical Software for Multiobjective Optimization Problems."
    Ph.D. dissertation, Virginia Tech, Dept. of Computer Science, 2020.

------------------------------------------------------------------------------

The full list of public classes in this module includes the 7 unconstrained
DTLZ problems
 * ``dtlz1``,
 * ``dtlz2``,
 * ``dtlz3``,
 * ``dtlz4``,
 * ``dtlz5``,
 * ``dtlz6``,
 * ``dtlz7``,
 * ``dtlz8``, and
 * ``dtlz9``

which are selected by setting the environment variable
``DEEPHYPER_BENCHMARK_DTLZ_PROB``.

## Usage

To use the benchmark follow this example set of instructions:

```python

# Set DTLZ problem environment variables before loading
import os
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = "5" # 5 vars
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = "3" # 2 objs
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "2" # DTLZ2 problem
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.6" # soln [x_o, .., x_n]=0.6

# Load & install DTLZ benchmark suite
import deephyper_benchmark as dhb
dhb.install("DTLZ")
dhb.load("DTLZ")

# Example of running one evaluation of DTLZ problem
from deephyper.evaluator import RunningJob
config = dtlz.hpo.problem.default_configuration # get a default config to test
res = dtlz.hpo.run(RunningJob(parameters=config))

```
