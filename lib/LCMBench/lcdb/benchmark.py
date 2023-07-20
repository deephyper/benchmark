from deephyper_benchmark import *


class LCDBBenchmark(Benchmark):

    version = "0.0.1"

    requires = {
        "py-lcdb": {"type": "pip", "name": "lcdb"},
        "py-func-timeout": {"type": "pip", "name": "func-timeout"},
    }
