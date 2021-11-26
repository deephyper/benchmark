import abc
import cProfile
import pstats
import time


class Benchmark(abc.ABC):
    native_parameters = {"profiling": False}

    def __init__(self, verbose=0) -> None:
        self.parameters = {**self.native_parameters, **self.parameters}
        self.verbose = verbose

    def load_parameters(self, params) -> dict:
        """Load, verify the consistency of given parameters and return them as a dict."""
        for param, value in params.items():
            if param in self.parameters.keys():
                self.parameters[param] = value

        return self.parameters

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the corresponding benchmark."""

    @abc.abstractmethod
    def execute(self) -> None:
        """Execute the corresponding benchmark."""

    @abc.abstractmethod
    def report(self) -> dict:
        """Compute the desired results and return them as a dict."""

    def run(self):
        self.results = {}

        init_start = time.perf_counter()
        self.initialize()
        init_end = time.perf_counter()

        if self.parameters["profiling"]:
            prof = cProfile.Profile()
            prof.enable()

        exec_start = time.perf_counter()
        self.execute()
        exec_end = time.perf_counter()

        if self.parameters["profiling"]:
            prof.disable()
            stat = pstats.Stats(prof)

            def _clear_queue(val):
                val[-1] = {}
                return val

            stats = [
                [list(func_path), _clear_queue(list(chronos))]
                for func_path, chronos in stat.stats.items()
            ]
            self.results["profiling"] = stats
        self.results["init_time"] = init_end - init_start
        self.results["exec_time"] = exec_end - exec_start
        return self.report()
