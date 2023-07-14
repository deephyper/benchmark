""" This module contains objective function implementations of the JAHS 201
benchmark suite, implemented as DeepHyper compatible models.

"""

import numpy as np

class jahs_bench:
    """ A callable class implementing the JAHS benchmark problems. """

    def __init__(self, dataset="fashion_mnist"):
        """ Import and configure the jahs-bench module.

        Args:
            dataset (str): One of "cifar10", "colorectal_history",
                or "fashion_mnist" (default)

        """

        from jahs_bench.api import Benchmark
        import os

        ### JAHS bench settings ###
        MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
        # Define the benchmark
        self.benchmark = Benchmark(
                task=dataset,
                save_dir=MODEL_PATH,
                kind="surrogate",
                download=False
            )

    def __call__(self, x):
        """ DeepHyper compatible objective function calling jahs-bench.

        Args:
            x (dict): Configuration dictionary with same keys as jahs-bench.

        Returns:
            tuple of floats: In order: accuracy (to be maximized), latency
            (to be minimized).

        """

        # Default config
        config = {
            'Optimizer': 'SGD',
            'N': 5,
            'W': 16,
            'Resolution': 1.0,
        }
        # Update config using x
        for key in x.keys():
            config[key] = x[key]
        # Special rule for setting "TrivialAugment"
        if x['TrivialAugment'] == "on":
            config['TrivialAugment'] = True
        else:
            config['TrivialAugment'] = False
        # Check for nepochs
        nepochs = 200
        if 'nepochs' in x.keys():
            nepochs = x['nepochs']
        # Evaluate and return
        fx = np.zeros(2)
        result = self.benchmark(config, nepochs=nepochs)
        return result[nepochs]
