"""Hartmann6D function benchmark.

Description of the function: https://www.sfu.ca/~ssurjano/hart6.html
"""

from . import hpo_benchmark

__all__ = ["install", "init", "hpo"]

hpo = hpo_benchmark.benchmark


def install(config: dict = None):
    """Install the benchmark."""


def init(config: dict = None):
    """Initialize the benchmark."""
