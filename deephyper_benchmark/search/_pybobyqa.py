import os

import ConfigSpace.hyperparameters as csh
import pandas as pd


from deephyper.core.exceptions import SearchTerminationError  # noqa: E402
from deephyper.core.utils._timeout import terminate_on_timeout  # noqa: E402
from deephyper.evaluator import RunningJob  # noqa: E402
from deephyper.search import Search  # noqa: E402

import pybobyqa


def pybobyqa_bounds_from_hp(cs_hp):
    if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
        lower, upper = float(cs_hp.lower), float(cs_hp.upper)
    elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
        lower, upper = cs_hp.lower, cs_hp.upper
    elif isinstance(cs_hp, csh.CategoricalHyperparameter):
        lower, upper = 0.0, float(len(cs_hp.choices) - 1)
    elif isinstance(cs_hp, csh.OrdinalHyperparameter):
        lower, upper = 0.0, float(len(cs_hp.sequence) - 1)
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")
    return lower, upper


def convert_problem_to_pybobyqa(problem):
    xl, xu = [], []
    for cs_hp in problem.space.get_hyperparameters():
        lower, upper = pybobyqa_bounds_from_hp(cs_hp)
        xl.append(lower)
        xu.append(upper)
    return xl, xu


class PyBOBYQA(Search):
    """Wrapper for PyBOBYQA derivative-free optimizater.

    $ pip install Py-BOBYQA

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

        self._xl, self._xu = convert_problem_to_pybobyqa(problem)

    def _search(self, max_evals, timeout):
        def objective_wrapper(x):
            config = {k: v for k, v in zip(self._problem.hyperparameter_names, x)}
            self._evaluator.submit([config])
            job = self._evaluator.gather("ALL")[0]
            _, y = job
            return -y

        x0 = self._random_state.uniform(low=self._xl, high=self._xu)

        pyboobyqa_results = pybobyqa.solve(
            objective_wrapper,
            x0=x0,
            bounds=(self._xl, self._xu),
            maxfun=max_evals,
            seek_global_minimum=True,
        )

        self._evaluator.dump_evals(log_dir=self._log_dir)
        df_path = os.path.join(self._log_dir, "results.csv")
        df_results = pd.read_csv(df_path)
        return df_results
