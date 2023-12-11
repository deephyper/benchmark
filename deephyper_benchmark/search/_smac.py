import functools
import logging
import os

import ConfigSpace.hyperparameters as csh
import pandas as pd


from deephyper.core.exceptions import SearchTerminationError  # noqa: E402
from deephyper.core.utils._timeout import terminate_on_timeout  # noqa: E402
from deephyper.evaluator import RunningJob  # noqa: E402
from deephyper.search import Search  # noqa: E402

import smac
import smac.acquisition.function
import smac.multi_objective.parego


MAP_acq_func = {"UCB": "LCB"}


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
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        acq_func: str = "UCB",
        acq_func_kwargs: dict = None,
        n_objectives: int = 1,
        n_initial_points: int = None,
        **kwargs,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        self._acq_func = MAP_acq_func.get(acq_func, acq_func)
        self._acq_func_kwargs = acq_func_kwargs if acq_func_kwargs is not None else {}
        self.n_objectives = n_objectives
        self.n_initial_points = n_initial_points if n_initial_points is not None else 10
        self.seed = self._random_state.randint(low=0, high=2**31)

    def _search(self, max_evals, timeout):
        def objective_wrapper(config, seed=None):
            self._evaluator.submit([dict(config)])
            job = self._evaluator.gather("ALL")[0]
            _, y = job
            if self.n_objectives > 1:
                y = {f"cost_{i}": -y[i] for i in range(self.n_objectives)}
            else:
                y = -y
            return y

        scenario = smac.Scenario(
            self._problem.space,
            objectives="cost"
            if self.n_objectives == 1
            else [f"cost_{i}" for i in range(self.n_objectives)],
            n_trials=max_evals,
            output_directory=os.path.join(self._log_dir, "smac_output"),
            seed=self.seed,
        )

        initial_design = smac.HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=self.n_initial_points
        )

        # Use SMAC to find the best configuration/hyperparameters
        optimizer = smac.HyperparameterOptimizationFacade(
            scenario,
            target_function=objective_wrapper,
            acquisition_function=getattr(smac.acquisition.function, self._acq_func)(
                **self._acq_func_kwargs
            ),
            initial_design=initial_design,
            multi_objective_algorithm=None
            if self.n_objectives == 1
            else smac.multi_objective.parego.ParEGO(scenario),
            logging_level=False,
        )
        incumbent = optimizer.optimize()

        self._evaluator.dump_evals(log_dir=self._log_dir)
        df_path = os.path.join(self._log_dir, "results.csv")
        df_results = pd.read_csv(df_path)
        return df_results
