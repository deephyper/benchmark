import numpy as np
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from .model import get_data, PINN, BurgerSupervisor

# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=5)
problem.add_hyperparameter((1e-5, 1e-2), "lr", default_value=0.01)
problem.add_hyperparameter((5, 50), "hidden_dim", default_value=5)
problem.add_hyperparameter((200, 300), "epochs", default_value=200)
problem.add_hyperparameter((0., 5.), "alpha", default_value=.5)
problem.add_hyperparameter(["relu", "leaky_relu", "tanh", "elu",
                            "gelu", "sigmoid"],
                            "activation", default_value='tanh')


# define the run function
def run(config):
    nu=0.01/np.pi
    train, val, test = get_data(NT=200, NX=128, X_U=1, X_L=-1, T_max=1,
                                Nu=nu)
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    output_dim = 1
    epochs = config["epochs"]
    lr = config["lr"]
    alpha = config["alpha"]
    activation = config["activation"]
    net = PINN(num_layers=num_layers, hidden_dim=hidden_dim,
             output_dim=output_dim, act_fn=activation)
    sup = BurgerSupervisor(nu, net, epochs, lr, alpha)
    val_f_loss = sup.train(train, val)
    return -val_f_loss


def evaluate(config):
    """
    Evaluate an hyperparameter configuration
    on training/validation and testing data.
    """
    nu = 0.01/np.pi
    train, val, test = get_data(NT=200, NX=100, X_U=1., X_L=-1., T_max=1, Nu=nu)
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    output_dim = 1
    epochs = config["epochs"]
    lr = config["lr"]
    alpha = config["alpha"]
    activation = config["activation"]
    net = PINN(num_layers=num_layers, hidden_dim=hidden_dim,
             output_dim=output_dim, act_fn=activation)
    sup = BurgerSupervisor(nu=nu, net=net, epochs=epochs, lr=lr, alpha=alpha)
    val_f_loss = sup.train(train, test) # evaluate the best config on the testing set.
