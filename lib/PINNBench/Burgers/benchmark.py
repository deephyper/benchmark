
import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class PINNBurgersBenchmark(Benchmark):

    version = "0.0.1"

    requires = {
        "bash-install": {"type": "cmd", "cmd": os.path.join(DIR, "../install.sh")}, # not sure what this line should do or I should do to match this line.
    }