import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class ShekelFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/shekel.html
    """

    version = "0.0.1"
