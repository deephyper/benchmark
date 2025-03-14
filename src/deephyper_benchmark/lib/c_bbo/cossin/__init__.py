"""Cos-Sin function benchmark.

Description of the function: ...
"""

from . import hpo_benchmark

__all__ = ["install", "init", "hpo"]

hpo = hpo_benchmark.benchmark


def install(config: dict = None):
    """Install the benchmark."""


def init(config: dict = None):
    """Initialize the benchmark."""
