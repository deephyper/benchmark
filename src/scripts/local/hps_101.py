from deephyper_benchmark.benchmark import Benchmark

class LocalHPS101Benchmark(Benchmark):

    def initialize(self):
        print(f"Initializing {type(self).__name__}")

    def execute(self):
        print(f"Executing {type(self).__name__}")

    def report(self):
        print(f"Reporting {type(self).__name__}")