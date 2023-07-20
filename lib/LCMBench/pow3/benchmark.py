import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class Pow3Benchmark(Benchmark):

    version = "0.0.1"

    requires = {}