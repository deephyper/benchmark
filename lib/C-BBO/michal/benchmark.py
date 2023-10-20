import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class MichalFunction(Benchmark):
    """https://www.sfu.ca/~ssurjano/michal.html
    """

    version = "0.0.1"
