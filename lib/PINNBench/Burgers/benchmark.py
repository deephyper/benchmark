import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class PINNBenchmark(Benchmark):

    version = "0.5.1"

    requires = {
        "makefile": {"type": "cmd", "cmd": "make build"},
        "py-astropy": {"type": "pip", "name": "astropy"},
        "py-patsy": {"type": "pip", "name": "patsy"},
        "py-statsmodels": {"type": "pip", "name": "statsmodels"},
        "pkg-candle": {"type": "pythonpath", "path": f"{DIR}/build/Benchmarks/common"},
        "pkg-combo": {"type": "pythonpath", "path": f"{DIR}/build/Benchmarks/Pilot1/Combo"},
    }
    