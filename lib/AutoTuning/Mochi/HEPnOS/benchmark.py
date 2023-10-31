import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class HEPnOS(Benchmark):
    """
    - Experimental data from the HEPnOS paper: https://github.com/hepnos/HEPnOS-Autotuning-analysis
    - Experimental code from the HEPnOS paper: https://github.com/hepnos/HEPnOS-Autotuning/tree/dev-new-hepnos
    - Paper from M. Dorier and R. Egele et al. "HPC Storage Service Autotuning Using Variational-Autoencoder-Guided Asynchronous Bayesian Optimization". https://arxiv.org/pdf/2210.00798
    """

    version = "0.0.1"

    requires = {
        "makefile": {"step": "install", "type": "cmd", "cmd": "make build"},
    }
