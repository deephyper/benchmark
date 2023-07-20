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

    def install(self) -> int:
        """Install the benchmark.

        Returns:
            int: 0 if the installation is successful, error code otherwise.
        """

        for rname, rcommand in self.requires.items():
            res = 0

            if "install" in rcommand["step"]:
                res = self.process_requirement(rname, rcommand)

            if res != 0:
                logging.error("Installation failed for requirement: {rname}")
                return res

    def load(self) -> int:
        """Loads the benchmark and returns the corresponding Python module."""

        for rname, rcommand in self.requires.items():
            res = 0

            if "load" in rcommand["step"]:
                res = self.process_requirement(rname, rcommand)

            if res != 0:
                logging.error("Loading failed for requirement: {rname}")
                return res

        module_name = f"deephyper_benchmark.lib.{self.name}"
        spec = importlib.util.spec_from_file_location(
            module_name, f"{self.cwd}/__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.module = module

        return self.module

    def process_requirement(self, rname: str, rcommand: dict) -> int:
        logging.info(f"Processing requirement: {rname}")

        res = 0

        # Add path to PYTHONPATH
        if rname["type"] == "pythonpath":
            path = rcommand["path"]
            logging.info(f"Adding {path} to PYTHONPATH")
            sys.path.insert(0, path)

        # Add/Update environment variable to environment.
        elif rcommand["type"] == "env":
            os.environ[rcommand["key"]] = rcommand["value"]

        elif rcommand["type"] == "cmd":
            cmd = rcommand["cmd"]
            res = os.system(f"cd {self.cwd} && {cmd}")
        elif rcommand["type"] == "pip":
            args = rcommand["args"]
            res = os.system(f"pip {args}")

        return res
