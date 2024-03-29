import json
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def array_to_json(x: np.ndarray) -> str:
    """Converts a numpy array to a json string.

    Args:
        x (np.ndarray): a numpy array.

    Returns:
        str: a json string.
    """
    x_json = json.dumps(x, cls=NumpyArrayEncoder)
    return x_json


def json_to_array(x_json: str) -> np.ndarray:
    """Converts an array saved as JSON to a numpy array.

    Args:
        x_json (str): the json encoded array.

    Returns:
        np.ndarray: the decoded numpy array.
    """
    x = np.asarray(json.loads(x_json))
    return x


if __name__ == "__main__":
    """Example of usage of array_to_json function."""
    x = np.array([1, 2, 3])
    print(x)
    print(type(x))
    print(array_to_json(x))
    print(type(array_to_json(x)))
