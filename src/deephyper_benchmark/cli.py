import argparse
import logging
import importlib.util
import os

from deephyper_benchmark.benchmark import Benchmark

logger = logging.getLogger(__name__)


def _create_parser():
    parser = argparse.ArgumentParser(description="DeepHyper/Benchmark command line.")
    parser.add_argument("script", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _run_benchmark(script, verbose):
    logger.info(f"Loading benchmark from: {script}")

    # mscript = importlib.import_module(script)
    # specify the module that needs to be
    # imported relative to the path of the
    # module
    spec = importlib.util.spec_from_file_location(os.path.basename(script)[:-3], script)

    # creates a new module based on spec
    mscript = importlib.util.module_from_spec(spec)

    # executes the module in its own namespace
    # when a module is imported or reloaded.
    spec.loader.exec_module(mscript)

    def is_valid_benchmark(cls):
        try:
            b = cls is not Benchmark and issubclass(cls, Benchmark)
        except:
            b =False
        return b

    benchmark_classes = [
        getattr(mscript, attr)
        for attr in dir(mscript)
        if is_valid_benchmark(getattr(mscript, attr))
    ]

    for benchmark_class in benchmark_classes:

        benchmark = benchmark_class()
        benchmark.initialize()
        benchmark.execute()
        benchmark.report()


def main():

    parser = _create_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

    _run_benchmark(**vars(args))


if __name__ == "__main__":
    main()
