import abc
import time


class Benchmark(abc.ABC):
    parameters: dict

    def __init__(self, verbose=0) -> None:
        super().__init__()
        self.verbose = verbose

    def load_parameters(self, params) -> dict:
        """Load, verify the consistency of given parameters and return them as a dict.
        """
        for param, value in params.items():
            if param in self.parameters.keys():
                self.parameters[param] = value
                
        return self.parameters

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the corresponding benchmark.
        """

    @abc.abstractmethod
    def execute(self) -> None:
        """Execute the corresponding benchmark.
        """

    @abc.abstractmethod
    def report(self) -> dict:
        """Compute the desired results and return them as a dict.
        """

    def run(self):
        self.results = {}

        init_start = time.time()
        self.initialize()
        init_end = time.time()
        exec_start = time.time()
        self.execute()
        exec_end = time.time()

        self.results["init_time"] = init_end - init_start
        self.results["exec_time"] = exec_end - exec_start
        return self.report()
