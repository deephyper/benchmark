import functools
import logging
import os

import ConfigSpace.hyperparameters as csh
import numpy as np
import pandas as pd


from deephyper.core.exceptions import SearchTerminationError  # noqa: E402
from deephyper.core.utils._timeout import terminate_on_timeout  # noqa: E402
from deephyper.evaluator import RunningJob  # noqa: E402
from deephyper.search import Search  # noqa: E402

import dehb.optimizers


MAP_acq_func = {"UCB": "LCB"}


class DEAutoML(Search):
    """Wrapper for DE optimizer from: https://github.com/automl/DEHB

    $ pip install dehb

    Papers:
        1. Awad, Noor, Neeratyoy Mallik, and Frank Hutter. "Differential evolution for neural architecture search." arXiv preprint arXiv:2012.06400 (2020). ICLR Workshop, 2020.
        2. Awad, Noor, Neeratyoy Mallik, and Frank Hutter. "Dehb: Evolutionary hyperband for scalable, robust and efficient hyperparameter optimization." arXiv preprint arXiv:2105.09821 (2021).



    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.
        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.
        random_state (int, optional): Random seed. Defaults to ``None``.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ``"."``.
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

    def _search(self, max_evals, timeout):
        def objective_wrapper(config, **kwargs):
            self._evaluator.submit([dict(config)])
            job = self._evaluator.gather("ALL")[0]
            _, y = job
            if len(self._evaluator.jobs_done) >= max_evals:
                raise SearchTerminationError("Maximum number of evaluations reached!")
            return {"fitness": y, "cost": 0}

        optimizer = dehb.optimizers.DE(
            f=objective_wrapper,
            cs=self._problem.space,
            # TODO: Here we followed the default parameters of DEHB available in `dehb==0.0.7`
            # https://github.com/automl/DEHB/blob/2d3d178df65ce4402dd8998e483386ad309ea315/src/dehb/optimizers/dehb.py#L170
            mutation_factor=0.5,
            crossover_prob=0.5,
            strategy="rand1_bin",
            pop_size=20,
            output_path=os.path.join(self._log_dir, "optimizer-output"),
            n_workers=1,
        )

        try:
            optimizer.run(
                generations=(max_evals // 20 + 1),
                verbose=False,
                save_intermediate=False,
            )
        except SearchTerminationError:
            pass

        self._evaluator.dump_evals(log_dir=self._log_dir)
        df_path = os.path.join(self._log_dir, "results.csv")
        df_results = pd.read_csv(df_path)
        return df_results
