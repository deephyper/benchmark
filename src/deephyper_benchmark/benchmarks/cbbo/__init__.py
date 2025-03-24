"""Subpackage for C-BBO benchmarks."""

from .ackley import AckleyBenchmark
from .branin import BraninBenchmark
from .easom import EasomBenchmark
from .griewank import GriewankBenchmark
from .hartmann6D import Hartmann6DBenchmark
from .levy import LevyBenchmark
from .michal import MichalBenchmark

__all__ = [
    "AckleyBenchmark",
    "BraninBenchmark",
    "EasomBenchmark",
    "GriewankBenchmark",
    "Hartmann6DBenchmark",
    "LevyBenchmark",
    "MichalBenchmark",
]
