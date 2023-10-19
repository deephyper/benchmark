import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class Hartmann6DFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/hart6.html
    """

    version = "0.0.1"
