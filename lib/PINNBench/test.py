import os
from  DiffusionReaction import hpo
from deephyper.evaluator import profile, RunningJob
from deephyper.stopper.integration import DeepXDEStopperCallback
from deephyper.stopper import LCModelStopper
from deephyper.search.hps import CBO

os.environ['DEEPHYPER_BENCHMARK_DATASET'] = '2D_diff-react_NA_NA'
default_config = hpo.problem.default_configuration
result = hpo.run(RunningJob(parameters=default_config))
print(f"{result=}")

stopper = LCModelStopper(min_steps=1, max_steps=200)
search = CBO(
        hpo.problem, hpo.run, initial_points=[hpo.problem.default_configuration], stopper=stopper
)
results = search.search(max_evals=1)