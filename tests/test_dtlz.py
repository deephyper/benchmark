"""here."""

from deephyper.hpo import RandomSearch
import deephyper_benchmark.benchmarks.dtlz as bench


def test_dtlz_benchmarks(tmp_path="."):
    """Test cbbo benchmarks."""

    max_evals = 25
    nobj = 2

    for prob_id in range(1, 8):
        b = bench.DTLZBenchmark(prob_id=prob_id, nobj=nobj)
        search = RandomSearch(b.problem, b.run_function, log_dir=tmp_path)
        results = search.search(max_evals)
        assert len(results) == max_evals
        cols = [f"objective_{i}" for i in range(nobj)]
        assert all(c in results.columns for c in cols)
        hvi = b.scorer.hypervolume(results[cols].values)
        assert len(hvi) == max_evals
