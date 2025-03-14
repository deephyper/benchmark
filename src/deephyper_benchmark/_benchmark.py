import abc


class Benchmark(abc.ABC):
    def __init__(self):
        super().__init__()
        self.refresh_settings()

    def refresh_settings(self):
        """Refresh benchmark settings."""


class HPOBenchmark(Benchmark):
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
