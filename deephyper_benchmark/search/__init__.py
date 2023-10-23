__all__ = []

try:
    from deephyper_benchmark.search._mpi_doptuna import (
        MPIDistributedOptuna,
    )  # noqa: F401

    __all__.append("MPIDistributedOptuna")
except ImportError:
    pass

try:
    from deephyper_benchmark.search._cobyqa import COBYQA  # noqa: F401

    __all__.append("COBYQA")
except ImportError:
    pass

try:
    from deephyper_benchmark.search._pybobyqa import PyBOBYQA  # noqa: F401

    __all__.append("PyBOBYQA")
except ImportError:
    pass

try:
    from deephyper_benchmark.search._smac import SMAC  # noqa: F401

    __all__.append("SMAC")
except ImportError:
    pass

try:
    from deephyper_benchmark.search._de_automl import DEAutoML  # noqa: F401

    __all__.append("DEAutoML")
except ImportError:
    pass
