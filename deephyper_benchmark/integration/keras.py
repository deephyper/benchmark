import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def count_params(model: tf.keras.Model) -> dict:
    """Evaluate the number of parameters of a Keras model.

    Args:
        model (tf.keras.Model): a Keras model.

    Returns:
        dict: a dictionary with the number of trainable ``"num_parameters_train"`` and 
        non-trainable parameters ``"num_parameters"``.
    """
    num_parameters_train = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    )
    num_parameters = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    )
    return {
        "num_parameters": num_parameters,
        "num_parameters_train": num_parameters_train,
    }
