from deephyper.sklearn.classifier import run_autosklearn1

def load_diabetes():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml(name="diabetes", version=1, return_X_y=True)
    return X, y

def run_diabetes(config):
    return run_autosklearn1(config, load_diabetes)