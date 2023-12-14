from typing import Tuple

import numpy as np
import openml
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Reference of HPOBench Paper: https://arxiv.org/pdf/2109.06716.pdf
# Reference of original paper introducign these benchmarks: https://arxiv.org/pdf/1905.04970.pdf
# Trying to reproduce as much as possible results/setup from the original paper

map_task_to_openmlid = {
    # https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants
    "navalpropulsion": "44969",
    # https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
    "parkinsonstelemonitoring": "4531",
    # https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
    "proteinstructure": "44963",
    # https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis
    "slicelocalization": "42973",
}

map_openmlid_to_task = {v: k for k, v in map_task_to_openmlid.items()}

map_task_to_default_target_attribute = {
    "navalpropulsion": ["gt_compressor_decay_state_coefficient"],
    "parkinsonstelemonitoring": ["motor_UPDRS", "total_UPDRS"],
    "proteinstructure": ["RMSD"],
    "slicelocalization": ["reference"],
}


def load_from_openml(dataset_id: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load dataset from openml library.

    Args:
        dataset_id (str): the identifier of the dataset to be loaded from the OpenML database.

    Raises:
        ValueError: if the dataset is not found.

    Returns:
        (np.ndarray, np.ndarray, dict): X, y, metadata where X is the input array, y is the target array, and metadata is a dictionary containing additional information about the data.
    """
    task_name = map_openmlid_to_task[dataset_id]
    default_target_attribute = map_task_to_default_target_attribute[task_name]

    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )

    X, y, categorical_indicator, _ = dataset.get_data()
    y = X[default_target_attribute]
    X = X.drop(default_target_attribute, axis=1)
    y = y.values

    # Drop constant features
    constant_features = []
    continuous_features = []
    for i in range(X.shape[1]):
        values, _ = np.unique(X.values[:, i], return_counts=True)
        if len(values) == 1:
            constant_features.append(list(X.columns)[i])
        elif len(values) >= 10:
            continuous_features.append(list(X.columns)[i])

    X = X.drop(constant_features, axis=1)

    # Split into train and test
    test_size = int(0.2 * len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=test_size, random_state=42
    )

    # Standard Normalization on features
    m = np.mean(X_train[continuous_features], axis=0)
    s = np.std(X_train[continuous_features], axis=0)
    X_train[continuous_features] = (X_train[continuous_features] - m) / s
    X_valid[continuous_features] = (X_valid[continuous_features] - m) / s
    X_test[continuous_features] = (X_test[continuous_features] - m) / s

    # Standard Normalization on targets
    m = np.mean(y_train, axis=0)
    s = np.std(y_train, axis=0)
    y_train = (y_train - m) / s
    y_valid = (y_valid - m) / s
    y_test = (y_test - m) / s

    return X_train.values, X_valid.values, X_test.values, y_train, y_valid, y_test


if __name__ == "__main__":
    # Some tests
    # task = "navalpropulsion"
    # task = "proteinstructure"
    task = "slicelocalization"
    print(f"Task: {task}")
    openmlid = map_task_to_openmlid[task]
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_from_openml(openmlid)

    print("X_train.shape", X_train.shape)
    print("X_valid.shape", X_valid.shape)
    print("X_test.shape", X_test.shape)
    print("y_train.shape", y_train.shape)
    print("y_valid.shape", y_valid.shape)
    print("y_test.shape", y_test.shape)

    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)

    # As the data are normalized these values are expected to be close to 1.0
    fold = ["train", "valid", "test"]
    for i, (X, y) in enumerate(
        zip([X_train, X_valid, X_test], [y_train, y_valid, y_test])
    ):
        print(f"Fold: {fold[i]}")
        y_pred = model.predict(X)
        print(f"  MSE: {mean_squared_error(y, y_pred)}")
