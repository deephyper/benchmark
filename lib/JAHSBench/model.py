""" This module contains objective function implementations of the JAHS 201
benchmark suite, implemented as DeepHyper compatible models.

"""

import numpy as np

class jahs_bench:
    """ A callable class implementing the JAHS benchmark problems. """

    def __init__(self, nepochs=200, dataset="cifar10"):
        """ Import and configure the jahs-bench module.

        Args:
            nepochs (int, optional): Number of training epochs to use,
                defaults to 200.

            dataset (str): One of "cifar10" (default), "colorectal_history",
                or "fashion_mnist"

        """

        from jahs_bench.api import Benchmark

        ### JAHS bench settings ###
        self.nepochs = nepochs
        MODEL_PATH = "."
        # Define the benchmark
        self.benchmark = Benchmark(
                task=dataset,
                save_dir=MODEL_PATH,
                kind="surrogate",
                download=True
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
        # Evaluate and return
        fx = np.zeros(2)
        result = self.benchmark(config, nepochs=self.nepochs)
        fx[0] = result[self.nepochs]['valid-acc']
        fx[1] = result[self.nepochs]['latency']
        return fx[0], fx[1]
