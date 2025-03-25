"""here."""

from deephyper.hpo import CBO
from deephyper_benchmark.benchmarks.cbbo import BraninBenchmark


def main():
    """Run example."""
    bench = BraninBenchmark()

    search = CBO(bench.problem, bench.run_function)
    results = search.search(25)

    print(results)

    cumul_regret = bench.scorer.cumul_regret(results.objective)
    print(cumul_regret)


if __name__ == "__main__":
    main()
