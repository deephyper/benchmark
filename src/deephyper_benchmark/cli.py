import argparse
import logging
from deephyper_benchmark.utils import run_config

logger = logging.getLogger(__name__)


def _create_parser():
    parser = argparse.ArgumentParser(
        description="DeepHyper/Benchmark command line.")
    parser.add_argument("config", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main():

    parser = _create_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

    run_config(args.config, args.verbose)


if __name__ == "__main__":
    main()
