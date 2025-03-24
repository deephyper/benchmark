import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class LCBench(Benchmark):
    version = "0.0.1"

    requires = {
        "makefile": {"step": "install", "type": "cmd", "cmd": "make build"},
        "lcdbench-api": {
            "step": "load",
            "type": "pythonpath",
            "path": f"{DIR}/../build/LCBench/",
        },
    }
