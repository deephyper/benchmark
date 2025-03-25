import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class LCDBHyperparameterOptimization(Benchmark):
    """LCDB 2.0
    """

    version = "0.0.1"
