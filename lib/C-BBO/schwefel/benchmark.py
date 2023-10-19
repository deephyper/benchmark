import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class SchwefelFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/schwef.html
    """

    version = "0.0.1"
