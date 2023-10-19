import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class AckleyFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/ackley.html
    """

    version = "0.0.1"
