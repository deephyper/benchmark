from deephyper.problem import HpProblem
from .model import PINN, BurgerSupervisor, sanityCheckData
from deephyper.search.hps import CBO

# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=5)
problem.add_hyperparameter((1e-5, 1e-2), "lr", default_value=0.01)
problem.add_hyperparameter((5, 50), "hidden_dim", default_value=5)
problem.add_hyperparameter((2, 1024), "batch_size", default_value=2)
problem.add_hyperparameter((200, 300), "epochs", default_value=200)
problem.add_hyperparameter((0., 5.), "alpha", default_value=.5)
problem.add_hyperparameter(["relu", "leaky_relu", "tanh", "elu", "gelu", "sigmoid"], "activation", default_value='tanh')


# define the run function
def run(config):
    x, y, val = sanityCheckData()
    num_layers = config["num_layers"] 
    hidden_dim = config["hidden_dim"]
    output_dim = 1
    nu = 0.01
    epochs = config["epochs"] 
    batch_size = config["batch_size"]
    lr = config["lr"]
    alpha = config["alpha"]
    activation = config["activation"]
    net = PINN(num_layers, hidden_dim, output_dim, activation)
    sup = BurgerSupervisor(nu, net, epochs, batch_size, lr, alpha)
    val_f_loss = sup.train(x, y, val)
    return -val_f_loss


def evaluate(config):
    """Evaluate an hyperparameter configuration on training/validation and testing data."""
    x, y, val = sanityCheckData()
    num_layers = config["num_layers"] 
    hidden_dim = config["hidden_dim"]
    output_dim = 1
    nu = 0.01
    epochs = config["epochs"] 
    batch_size = config["batch_size"]
    lr = config["lr"]
    alpha = config["alpha"]
    activation = config["activation"] 
    net = PINN(num_layers, hidden_dim, output_dim, activation)
    sup = BurgerSupervisor(nu, net, epochs, batch_size, lr, alpha)
    val_f_loss = sup.train(x, y, val)

    