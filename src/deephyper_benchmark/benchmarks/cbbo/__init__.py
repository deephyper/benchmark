"""Subpackage for the C-BBO benchmarks."""

from .ackley import AckleyBenchmark
from .branin import BraninBenchmark
from .easom import EasomBenchmark
from .griewank import GriewankBenchmark
from .hartmann6D import Hartmann6DBenchmark
from .levy import LevyBenchmark
from .michal import MichalBenchmark
from .rosen import RosenBenchmark
from .schwefel import SchwefelBenchmark
from .shekel import ShekelBenchmark

__all__ = [
    "AckleyBenchmark",
    "BraninBenchmark",
    "EasomBenchmark",
    "GriewankBenchmark",
    "Hartmann6DBenchmark",
    "LevyBenchmark",
    "MichalBenchmark",
    "RosenBenchmark",
    "SchwefelBenchmark",
    "ShekelBenchmark",
]
