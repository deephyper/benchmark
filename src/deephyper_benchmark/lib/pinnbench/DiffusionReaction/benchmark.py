import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class PINNDiffusionReactionBenchmark(Benchmark):
    version = "0.0.1"

    requires = {
        "py-pip-requirements": {
            "step": "install",
            "type": "pip",
            "args": "install -r " + os.path.join(DIR, "requirements.txt"),
        },
        "bash-install": {
            "step": "install",
            "type": "cmd",
            "cmd": "cd .. && bash " + os.path.join(DIR, "./install.sh"),
        },
        "deepxde-backend": {
            "step": "load",
            "type": "env",
            "key": "DDE_BACKEND",
            "value": "pytorch",
        },
    }
