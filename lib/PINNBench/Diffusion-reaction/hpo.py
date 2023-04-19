from pdebench.models.pinn.train import *

from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import profile, RunningJob


@profile
def run(job: RunningJob) -> dict:
    config = job.parameters
    min_b = 1
    max_b = 1000
    val_loss, test_loss = run_training(
                            scenario="diff-react",
                            epochs=config['epochs'],
                            learning_rate=config['lr'],
                            model_update=500,
                            root_path='Users/yixuan/Documents/AllWorkStuff/DeepHyper/PDEBench/pdebench/data/2D_diff-react_NA_NA',
                            flnm="2D_diff-react_NA_NA.h5",
                            config=config,
                            seed="0000",
                            )
    objective = -val_loss
    metadata = {
        "num_hyperparameters": len(job.parameters),
        "val_loss": eval,
        "test_loss": test_loss,
    }
    return {"objective": objective, "metadata": metadata}


# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=5)
problem.add_hyperparameter((1e-5, 1e-2), "lr", default_value=0.01)
problem.add_hyperparameter((5, 50), "num_neurons", default_value=5)
problem.add_hyperparameter((1,200), "epochs", default_value=1)
problem.add_hyperparameter(
    ["relu", "leaky_relu", "tanh", "elu", "gelu", "sigmoid"],
    "activation",
    default_value="tanh",
)


def evaluate(config):
    """
    Evaluate an hyperparameter configuration
    on training/validation and testing data.
    """
    val_loss, test_loss = run_training(
                        scenario="diff-react",
                        epochs=config['epochs'],
                        learning_rate=config['lr'],
                        model_update=500,
                        root_path='Users/yixuan/Documents/AllWorkStuff/DeepHyper/PDEBench/pdebench/data/2D_diff-react_NA_NA',
                        flnm="2D_diff-react_NA_NA.h5",
                        config=config,
                        seed="0000",
                        )
    return test_loss




if __name__ == '__main__':
    default_config = problem.default_configuration
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")