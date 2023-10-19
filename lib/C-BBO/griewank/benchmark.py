import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class GriewankFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/griewank.html
    """

    version = "0.0.1"
