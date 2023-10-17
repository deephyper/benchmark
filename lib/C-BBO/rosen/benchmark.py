import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class RosenbrockFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/rosen.html
    """

    version = "0.0.1"
