import abc
import time

class Benchmark(abc.ABC):

    def __init__(self, verbose=0) -> None:
        super().__init__()

        self.results = {}
        self.verbose = verbose
    
    @abc.abstractmethod
    def load_parameters(self, **kwargs):
        """Load and verify the consistency of given parameters.
        """

    @abc.abstractmethod
    def initialize(self):
        """Initialize the corresponding benchmark.
        """

    @abc.abstractmethod
    def execute(self):
        """Execute the corresponding benchmark.
        """

    @abc.abstractmethod
    def report(self):
        """Report the results from the benchmark.
        """
    
    def run(self):
        start_init = time.time()
        self.initialize()
        end_init = time.time()
        start_exec = time.time()
        self.execute()
        end_exec = time.time()

        self.results["init_time"] = end_init - start_init
        self.results["exec_time"] = end_exec - start_exec
