import abc


class Benchmark(abc.ABC):
    """Base class for benchmarks."""


class HPOBenchmark(Benchmark):
    """Base class for Hyperparameter optimization benchmarks."""

    @property
    @abc.abstractmethod
    def problem(self):
        """The benchmark hyperparameter problem."""

    @property
    @abc.abstractmethod
    def run_function(self):
        """The benchmark hyperparameter run_function."""

    @property
    @abc.abstractmethod
    def scorer(self):
        """The benchmark hyperparameter scorer."""
