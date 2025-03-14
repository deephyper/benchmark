import os
from typing import Union, List, Dict

import ConfigSpace.hyperparameters as csh
import optuna
import pandas as pd
from optuna.trial import FrozenTrial, TrialState, Trial
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters
from deephyper.evaluator import HPOJob
from deephyper.hpo import Search
from deephyper.hpo.utils import get_inactive_value_of_hyperparameter


def optuna_suggest_from_hp(trial: Trial, hp: Hyperparameter):
    name = hp.name
    if isinstance(hp, csh.UniformIntegerHyperparameter):
        value = trial.suggest_int(name, hp.lower, hp.upper, log=hp.log)
    elif isinstance(hp, csh.UniformFloatHyperparameter):
        value = trial.suggest_float(name, hp.lower, hp.upper, log=hp.log)
    elif isinstance(hp, csh.CategoricalHyperparameter):
        value = trial.suggest_categorical(name, hp.choices)
    elif isinstance(hp, csh.OrdinalHyperparameter):
        value = trial.suggest_categorical(name, hp.sequence)
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(hp)}")

    return value


def optuna_suggest_from_configspace(trial: Trial, config_space: ConfigurationSpace) -> dict:
    config = {}
    for name in config_space:
        value = optuna_suggest_from_hp(trial, config_space[name])
        config[name] = value

    config = dict(deactivate_inactive_hyperparameters(config, config_space))

    for name in config_space:
        # If the parameter is inactive due to some conditions then we attribute the
        # lower bound value to break symmetries and enforce the same representation.
        if name not in config:
            config[name] = get_inactive_value_of_hyperparameter(config_space[name])

    return config


class CheckpointSaverCallback:
    def __init__(self, log_dir=".", states=(TrialState.COMPLETE,)) -> None:
        self._log_dir = log_dir
        self._states = states

    def __call__(self, study: optuna.study.Study, trial: FrozenTrial) -> None:
        all_trials = study.get_trials(deepcopy=False, states=self._states)
        # n_complete = len(all_trials)

        pd.DataFrame([t.user_attrs["results"] for t in all_trials]).to_csv(
            os.path.join(self._log_dir, "results.csv")
        )


# Constraints
def constraints(trial):
    return trial.user_attrs["constraints"]


# Supported samplers
supported_samplers = ["TPE", "CMAES", "NSGAII", "DUMMY", "BOTORCH", "QMC"]
supported_pruners = ["NOP", "SHA", "HB", "MED"]


# TODO: CMAES requires: $ pip install cmaes
# TODO: GP requires: $ pip install torch

class OptunaSearch(Search):
    """Wrapper for Optuna to run distributed optimization with MPI.

    Args:
        problem (HpProblem):
            Hyperparameter problem describing the search space to explore.

        evaluator (Evaluator):
            An ``Evaluator`` instance responsible of distributing the tasks.

        random_state (int, optional):
            Random seed. Defaults to ``None``.

        log_dir (str, optional):
            Log directory where search's results are saved. Defaults to ``"."``.

        verbose (int, optional):
            Indicate the verbosity level of the search. Defaults to ``0``.

        sampler (Union[str, optuna.samplers.BaseSampler], optional):
            Optimization strategy to suggest new hyperparameter configurations. Defaults
            to ``"TPE"``.

        pruner (Union[str, optuna.pruners.BasePruner], optional):
            Pruning strategy to perform early discarding of unpromizing configuraitons.
            Defaults to ``None``.

        n_objectives (int, optional): Number of objectives to optimize. Defaults to ``1``.

        study_name (str, optional):
            Name of the study in the database used by Optuna. Defaults to ``None``.

        storage (Union[str, optuna.storages.BaseStorage], optional):
            Database used by Optuna. Defaults to ``None``.

        checkpoint (bool, optional):
            If results should be checkpointed regularly to the ``log_dir``. Defaults to ``True``.

        moo_lower_bounds ([type], optional): [description]. Defaults to ``None``.

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        ValueError: _description_
        TypeError: _description_
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        sampler: Union[str, optuna.samplers.BaseSampler] = "TPE",
        pruner: Union[str, optuna.pruners.BasePruner] = None,
        n_objectives: int = 1,
        study_name: str = None,
        storage: Union[str, optuna.storages.BaseStorage] = None,
        checkpoint: bool = True,
        n_initial_points: int = None,
        moo_lower_bounds=None,
        **kwargs,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        optuna.logging.set_verbosity(
            optuna.logging.DEBUG if self._verbose else optuna.logging.ERROR
        )

        self._evaluator = evaluator

        # get the __init__ parameters
        _init_params = locals()

        self._n_initial_points = (
            2 * len(self._problem) if n_initial_points is None else n_initial_points
        )

        # Constraints
        self._moo_lower_bounds = moo_lower_bounds
        self._constraints_func = None
        if moo_lower_bounds is not None:
            if len(moo_lower_bounds) == n_objectives:
                self._constraints_func = constraints
            else:
                raise ValueError(
                    f"moo_lower_bounds should be of length {n_objectives} but is of length "
                    f"{len(moo_lower_bounds)}"
                )

        # Setup the sampler
        if isinstance(sampler, optuna.samplers.BaseSampler):
            pass
        elif isinstance(sampler, str):
            sampler_seed = self._random_state.randint(2**31)
            if sampler == "TPE":
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=self._n_initial_points,
                    seed=sampler_seed,
                    constraints_func=self._constraints_func,
                )
            elif sampler == "GP":
                sampler = optuna.samplers.GPSampler(
                    n_startup_trials=self._n_initial_points,
                    seed=sampler_seed,
                    constraints_func=self._constraints_func,
                )
            elif sampler == "CMAES":
                sampler = optuna.samplers.CmaEsSampler(
                    n_startup_trials=self._n_initial_points, seed=sampler_seed
                )
            elif sampler == "NSGAII":
                sampler = optuna.samplers.NSGAIISampler(
                    seed=sampler_seed,
                    constraints_func=self._constraints_func,
                )
            elif sampler == "DUMMY":
                sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
            elif sampler == "BOTORCH":
                from optuna.integration import BoTorchSampler

                sampler = BoTorchSampler(
                    n_startup_trials=self._n_initial_points,
                    seed=sampler_seed,
                    constraints_func=self._constraints_func,
                )
            elif sampler == "QMC":
                sampler = optuna.samplers.QMCSampler(seed=sampler_seed)

            else:
                raise ValueError(
                    f"Requested unknown sampler {sampler} should be one of {supported_samplers}"
                )
        else:
            raise TypeError(
                f"Sampler is of type {type(sampler)} but must be a str or "
                "optuna.samplers.BaseSampler!"
            )

        self._n_objectives = n_objectives

        # Setup the pruner
        if self._n_objectives > 1 or pruner is None:
            pruner = None
            if pruner is not None:
                raise ValueError("Multi-objective optimization does not support pruning!")
        else:
            if isinstance(pruner, optuna.pruners.BasePruner):
                pass
            elif isinstance(pruner, str):
                if pruner == "NOP":
                    pruner = optuna.pruners.NopPruner()
                elif pruner == "SHA":
                    pruner = optuna.pruners.SuccessiveHalvingPruner()
                elif pruner == "HB":
                    pruner = optuna.pruners.HyperbandPruner()
                elif pruner == "MED":
                    pruner = optuna.pruners.MedianPruner()
                else:
                    raise ValueError(
                        f"Requested unknown pruner {pruner} should be one of {supported_pruners}"
                    )
            else:
                raise TypeError(
                    f"Pruner is of type {type(pruner)} but must be a str or "
                    "optuna.pruners.BasePruner!"
                )
        self.pruner = pruner

        self._checkpoint = checkpoint

        study_params = dict(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

        if self._n_objectives > 1:
            study_params["directions"] = ["maximize" for _ in range(self._n_objectives)]
        else:
            study_params["direction"] = "maximize"

        self.timestamp = None

        self.study = optuna.create_study(**study_params)

        self._init_params = _init_params

        self._trials = {}
        self._trials_count = 0

    def _ask(self, n: int = 1) -> List[Dict]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """
        new_samples = []
        for i in range(n):
            trial = self.study.ask()
            new_sample = optuna_suggest_from_configspace(trial, self._problem.space)
            new_samples.append(new_sample)
            self._trials[self._trials_count] = trial
            self._trials_count += 1

        return new_samples

    def _tell(self, results: List[HPOJob]):
        """Tell the search the results of the evaluations.

        Args:
            results (List[HPOJob]): a dictionary containing the results of the evaluations.
        """
        for job in results:
            i = int(job.id.split(".")[-1])
            obj = job.objective
            # Do not add failures to population
            if isinstance(obj, str):
                obj = TrialState.FAIL

            trial = self._trials.pop(i)
            self.study.tell(trial, obj)
