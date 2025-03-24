"""here."""

import numpy as np
from deephyper.hpo import CBO
from deephyper_benchmark.benchmarks.dtlz import DTLZBenchmark


def main():
    """Run example."""
    nobj = 2
    bench = DTLZBenchmark(nobj=nobj)

    search = CBO(bench.problem, bench.run_function, acq_optimizer="sampling")
    results = search.search(25)

    print(results)

    obj = results[[f"objective_{i}" for i in range(nobj)]].values
    print(np.shape(obj))
    hvi = bench.scorer.hypervolume(obj)
    print(hvi)

    hvi_iter = bench.scorer.hypervolume_iter(obj)
    print(hvi_iter)


if __name__ == "__main__":
    main()
