import abc

class Benchmark(abc.ABC):

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

