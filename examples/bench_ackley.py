"""here."""

from deephyper.hpo import CBO
from deephyper_benchmark import AckleyBenchmark


def main():
    """Run example."""

    bench = AckleyBenchmark()

    search = CBO(bench.problem, bench.run_function)
    results = search.search(10)

    print(results)


if __name__ == "__main__":
    main()
