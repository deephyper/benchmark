import numpy as np

from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from .model import get_data, PINN, BurgerSupervisor, plotter
from deephyper.stopper import LCModelStopper
from deephyper.evaluator import profile, RunningJob


@profile
def run(job: RunningJob) -> dict:
    # load data
    nu = 0.01 / np.pi
    train, val, test = get_data(NT=200, NX=128, X_U=1, X_L=-1, T_max=1, Nu=nu)

    config = job.parameters
    min_b = 1
    max_b = 1000

    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    output_dim = 1
    epochs = 1  #
    lr = config["lr"]
    alpha = config["alpha"]
    activation = config["activation"]
    net = PINN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        act_fn=activation,
    )
    sup = BurgerSupervisor(nu, net, epochs, lr, alpha)

    for budget_i in range(min_b, max_b + 1):
        eval = sup.step(train, val)
        print("val mse", eval)
        objective_i = -eval  # maximizing in deephyper
        job.record(budget_i, objective_i)
        if job.stopped():
            break
    objective = job.objective

    metadata = {
        "budget": budget_i,
        "stopped": budget_i < max_b,
        "infos_stopped": job.stopper.infos_stopped
        if hasattr(job, "stopper") and hasattr(job.stopper, "infos_stopped")
        else None,
    }
    return {"objective": objective, "metadata": metadata}


# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=5)
problem.add_hyperparameter((1e-5, 1e-2), "lr", default_value=0.01)
problem.add_hyperparameter((5, 50), "hidden_dim", default_value=5)
problem.add_hyperparameter((0.0, 5.0), "alpha", default_value=0.5)
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
    nu = 0.01 / np.pi
    train, val, test = get_data(NT=200, NX=100, X_U=1.0, X_L=-1.0, T_max=1, Nu=nu)
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    output_dim = 1
    epochs = config["epochs"]
    lr = config["lr"]
    alpha = config["alpha"]
    activation = config["activation"]
    net = PINN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        act_fn=activation,
    )
    sup = BurgerSupervisor(nu=nu, net=net, epochs=epochs, lr=lr, alpha=alpha)
    val_f_loss = sup.train(train, test)  # evaluate the best config on the testing set.


if __name__ == "__main__":
    import time

    start_time = time.time()
    stopper = LCModelStopper(max_steps=1000, lc_model="mmf4")
    scheduler = {
        "type": "periodic-exp-decay",
        "periode": 25,
        "rate": 0.1,
    }

    default_config = problem.default_configuration
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
