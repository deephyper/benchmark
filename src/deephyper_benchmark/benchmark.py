import abc

class Benchmark(abc.ABC):

    def __init__(self, verbose=0) -> None:
        super().__init__()

        self.verbose = verbose

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

