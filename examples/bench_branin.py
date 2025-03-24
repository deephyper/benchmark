"""here."""

from deephyper.hpo import CBO
from deephyper_benchmark import BraninBenchmark


def main():
    """Run example."""

    bench = BraninBenchmark()

    search = CBO(bench.problem, bench.run_function)
    results = search.search(10)

    print(results)


if __name__ == "__main__":
    main()
