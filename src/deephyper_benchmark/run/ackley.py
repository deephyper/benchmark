import numpy as np
from numpy.core.fromnumeric import shape
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from deephyper.benchmark.benchmark_functions_wrappers import ackley_


def load_data(dim=10, size=100):
    """
    Generate data for polynome_2 function.
    Returns Tuple of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    prop = 0.80
    f, (a, b), _ = ackley_()
    d = b - a
    x = np.array([a + np.random.random(dim) * d for i in range(size)])
    y = np.array([[f(v)] for v in x])

    sep_index = int(prop * size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    return (train_X, train_y), (valid_X, valid_y)


def run_ackley(config):
    (train_X, train_y), (valid_X, valid_y) = load_data()

    input_l = Input(shape=shape(train_X[0]))
    x = Dense(config["units"], activation=config["activation"])(input_l)
    x = Dropout(config["dropout_rate"])(x)
    output = Dense(1, activation="linear")(x)
    model = Model(input_l, output)

    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=[RootMeanSquaredError()])

    history = model.fit(
        train_X, train_y, epochs=config["num_epochs"], validation_data=(valid_X, valid_y), verbose=0
    )

    return -history.history["val_root_mean_squared_error"][-1]