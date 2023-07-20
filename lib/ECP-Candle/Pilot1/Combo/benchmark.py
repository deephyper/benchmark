import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class ECPCandlePilot1Combo(Benchmark):
    version = "0.5.1"

    requires = {
        "makefile": {"step": "install", "type": "cmd", "cmd": "make build"},
        "py-pip-requirements": {
            "step": "install",
            "type": "pip",
            "args": "install -r " + os.path.join(DIR, "requirements.txt"),
        },
        "pkg-candle": {
            "step": "load",
            "type": "pythonpath",
            "path": f"{DIR}/build/Benchmarks/common",
        },
        "pkg-combo": {
            "step": "load",
            "type": "pythonpath",
            "path": f"{DIR}/build/Benchmarks/Pilot1/Combo",
        },
    }
