import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class JAHS201Benchmark(Benchmark):

    version = "0.0.1"
    requires = {
        "py-pip-requirements": {
            "type": "pip",
            "name": "-r " + os.path.join(DIR, "REQUIREMENTS.txt"),
        },
        "bash-install": {
            "type": "cmd",
            "cmd": "cd . && bash " + os.path.join(DIR, "./install.sh"),
        },
    }
