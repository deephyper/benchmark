import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class BraninFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/branin.html
    """

    version = "0.0.1"
