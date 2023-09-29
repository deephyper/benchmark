import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class AckleyFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/ackley.html#:~:text=The%20Ackley%20function%20is%20widely,large%20hole%20at%20the%20centre.
    """

    version = "0.0.1"
