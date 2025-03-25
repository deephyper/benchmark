"""Test module for CBBO benchmark functions."""

from deephyper.hpo import RandomSearch

import deephyper_benchmark.benchmarks.cbbo as bench


def test_cbbo_benchmarks(tmp_path="."):
    """Test cbbo benchmarks."""

    benchmarks = [
        bench.AckleyBenchmark(),
        bench.BraninBenchmark(),
        bench.EasomBenchmark(),
        bench.GriewankBenchmark(),
        bench.Hartmann6DBenchmark(),
        bench.LevyBenchmark(),
        bench.MichalBenchmark(),
        bench.RosenBenchmark(),
        bench.SchwefelBenchmark(),
        bench.ShekelBenchmark(),
    ]

    max_evals = 25
    for b in benchmarks:
        search = RandomSearch(b.problem, b.run_function, log_dir=tmp_path)
        results = search.search(max_evals)
        assert len(results) == max_evals
        assert hasattr(b.scorer, "nparams")
        assert hasattr(b.scorer, "x_max")
        assert hasattr(b.scorer, "y_max")
        cumul_regret = b.scorer.cumul_regret(results.objective)
        assert len(cumul_regret) == max_evals
