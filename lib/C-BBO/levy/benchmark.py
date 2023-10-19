import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class LevyFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/levy.html
    """

    version = "0.0.1"
