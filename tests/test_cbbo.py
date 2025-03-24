"""here."""

from deephyper.hpo import RandomSearch
import deephyper_benchmark.benchmarks.cbbo as bench


def test_cbbo_benchmarks():
    """Test cbbo benchmarks."""

    benchmarks = [
        bench.AckleyBenchmark(),
        bench.BraninBenchmark(),
        bench.EasomBenchmark(),
        bench.GriewankBenchmark(),
        bench.Hartmann6DBenchmark(),
    ]

    max_evals = 25
    for b in benchmarks:
        search = RandomSearch(b.problem, b.run_function)
        results = search.search(max_evals)
        assert len(results) == max_evals
        cumul_regret = b.scorer.cumul_regret(results.objective)
        assert len(cumul_regret) == max_evals
