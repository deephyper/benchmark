import numpy as np
from deephyper.sklearn.classifier import run_autosklearn1


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)


def run_ackley(config):
    x = np.array([config["x"]])
    return -ackley(x)


def load_data():
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(name="diabetes", version=1, return_X_y=True)

    return X, y


def run_diabetes(config):
    if type(config["n_neighbors"]) == float:
        config['n_neighbors'] = int(config['n_neighbors'])
    return run_autosklearn1(config, load_data)
