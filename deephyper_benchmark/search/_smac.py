import functools
import logging
import os

import ConfigSpace.hyperparameters as csh
import pandas as pd


from deephyper.core.exceptions import SearchTerminationError  # noqa: E402
from deephyper.core.utils._timeout import terminate_on_timeout  # noqa: E402
from deephyper.evaluator import RunningJob  # noqa: E402
from deephyper.search import Search  # noqa: E402

from smac import HyperparameterOptimizationFacade, Scenario


class SMAC(Search):
    """Wrapper for SMAC optimizer.

    $ conda install gxx_linux-64 gcc_linux-64 swig
    $ pip install smac

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.
        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.
        random_state (int, optional): Random seed. Defaults to ``None``.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ``"."``.
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.
        deterministic (bool, optional): If ``True`` SMAC will be deterministic. Defaults to ``True``.
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
        def objective_wrapper(config, seed=None):
            self._evaluator.submit([dict(config)])
            job = self._evaluator.gather("ALL")[0]
            _, y = job
            return -y

        scenario = Scenario(
            self._problem.space,
            n_trials=max_evals,
            output_directory=os.path.join(self._log_dir, "smac_output"),
        )

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, objective_wrapper)
        incumbent = smac.optimize()

        self._evaluator.dump_evals(log_dir=self._log_dir)
        df_path = os.path.join(self._log_dir, "results.csv")
        df_results = pd.read_csv(df_path)
        return df_results
