import importlib
import importlib.util
import logging
import os
import sys

# Path of the root directory of the deephyper_benchmark package
PKG_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path of the root directory of the library of benchmarks
BCH_ROOT_DIR = os.path.join(os.path.dirname(PKG_ROOT_DIR), "lib")


def find_benchmark(benchmark_name: str):
    """Load a benchmark class from its string path.

    Args:
        benchmark_name (str): string path of the benchmark to load.

    Returns:
        Benchmark: the instanciated benchmark class.
    """

    benchmark_path = os.path.join(BCH_ROOT_DIR, benchmark_name)
    sys.path.insert(0, benchmark_path)
    # looking for "benchmark.py" in the benchmark folder
    benchmark_module = importlib.import_module("benchmark")

    benchmark_class = None
    for attr_name in benchmark_module.__dict__:
        attr = getattr(benchmark_module, attr_name)
        try:
            if issubclass(attr, Benchmark) and not (attr is Benchmark):
                benchmark_class = attr
                break
        except TypeError:
            pass

    if benchmark_class is None:
        logging.error("Cannot find any benchmark")
        return
    else:
        logging.info(f"Found {attr.__name__}")

    benchmark = benchmark_class()

    # Working directory of the benchmark
    benchmark.cwd = benchmark_path

    # Python module name of the benchmark
    benchmark.name = benchmark_name.replace("-", "_").replace("/", ".").lower()

    return benchmark


def install(name: str):
    """Installing benchmark from string path name. The installation is executed from the root directory of the
    requested benchmark."""

    logging.info(f"Installing {name}")

    benchmark = find_benchmark(name)
    benchmark.install()

    logging.info(f"Installation done")


def load(name: str):
    """Loading benchmark from string path name."""
    logging.info(f"Loading {name}")

    benchmark = find_benchmark(name)
    module = benchmark.load()

    logging.info(f"Loading done")

    return module


class Benchmark:
    """Class representing a benchmark."""

    requires = {}

    def __init__(self):
        self.name = None
        self.cwd = None

    def install(self):
        """Runs the installation of the benchmark."""

        for require_name, require_val in self.requires.items():
            res = 0
            if require_val["type"] == "cmd":
                logging.info(f"running requires[{require_name}]")
                cmd = require_val["cmd"]
                res = os.system(f"cd {self.cwd} && {cmd}")
            elif require_val["type"] == "pip":
                name = require_val["name"]
                res = os.system(f"pip install {name}")

            if res != 0:
                logging.error("installation failed!")

    def load(self):
        """Loads the benchmark and returns the corresponding Python module."""

        for require_name, require_val in self.requires.items():
            if require_val["type"] == "pythonpath":
                path = require_val["path"]
                logging.info(f"Adding {path} to PYTHONPATH")
                sys.path.insert(0, path)
            elif require_val["type"] == "env":
                os.environ[require_val["key"]] = require_val["value"]

        module_name = f"deephyper_benchmark.lib.{self.name}"
        spec = importlib.util.spec_from_file_location(
            module_name, f"{self.cwd}/__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.module = module

        return self.module
