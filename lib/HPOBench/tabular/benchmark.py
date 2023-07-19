import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class HPOBenchTabularNavalPropulsion(Benchmark):

    version = "0.0.1"

    requires = {
        "bash-install": {"type": "cmd", "cmd": os.path.join(DIR, "install.sh")},
    }
